"""Baseline credit-risk model: training, scoring, FI, sensitivity and performance.

The model is trained once on the labelled historical baseline and is never
retrained automatically. It supplies the model-specific terms of DIS:

- FI (SPEC 11.1): mean |SHAP| on the production batch (TreeSHAP), one-hot
  columns summed back to their source feature, scaled by the maximum.
- S  (SPEC 11.2): permutation-style sensitivity of predicted default
  probability to each feature within the production batch, scaled by the max.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from app.core.config import settings
from app.services.datasets import (
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    MONITORED_FEATURES,
    NUMERICAL_FEATURES,
)

logger = logging.getLogger(__name__)

ALGORITHM = "xgboost"
MIN_TRAIN_LABELS = 100


class ModelUnavailableError(ValueError):
    """Raised when the baseline cannot support a supervised model."""


def _features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[MONITORED_FEATURES].copy()
    for col in NUMERICAL_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce").astype(float)
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
    return x


def _build_pipeline(scale_pos_weight: float, seed: int) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERICAL_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=2,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def classification_metrics(
    y: np.ndarray, prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC and confusion matrix."""
    pred = (prob >= threshold).astype(int)
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "actual_default_rate": float(np.mean(y)),
        "predicted_default_rate": float(np.mean(pred)),
        "mean_predicted_probability": float(np.mean(prob)),
        "n": len(y),
    }


def split_labelled(
    df: pd.DataFrame, seed: int = settings.random_seed, test_size: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Deterministic stratified train/holdout split of the labelled rows."""
    labelled = df[df[LABEL_COLUMN].notna()]
    y = labelled[LABEL_COLUMN].astype(int).to_numpy()
    if len(y) < MIN_TRAIN_LABELS or len(np.unique(y)) < 2:
        raise ModelUnavailableError(
            f"Baseline needs at least {MIN_TRAIN_LABELS} labelled records with both "
            "default and non-default outcomes to train a credit-risk model."
        )
    x = _features(labelled)
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=seed)


def fit_pipeline(x_tr: pd.DataFrame, y_tr: np.ndarray, seed: int) -> Pipeline:
    """Fit the preprocessing + XGBoost pipeline with class-imbalance weighting."""
    pos = max(int(y_tr.sum()), 1)
    model = _build_pipeline((len(y_tr) - pos) / pos, seed)
    model.fit(x_tr, y_tr)
    return model


def select_threshold(model: Pipeline, x_tr: pd.DataFrame, y_tr: np.ndarray) -> float:
    """F1-maximising decision threshold chosen on the *training* split only."""
    train_prob = model.predict_proba(x_tr)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_tr, (train_prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def train_baseline_model(
    baseline: pd.DataFrame, seed: int = settings.random_seed
) -> tuple[Pipeline, dict[str, Any]]:
    """Train the champion model on the labelled baseline and evaluate on a holdout.

    The decision threshold maximises F1 on the training split (not the holdout),
    so holdout metrics stay an honest estimate.
    """
    x_tr, x_te, y_tr, y_te = split_labelled(baseline, seed)
    model = fit_pipeline(x_tr, y_tr, seed)

    threshold = select_threshold(model, x_tr, y_tr)

    hold_prob = model.predict_proba(x_te)[:, 1]
    metrics = classification_metrics(y_te, hold_prob, threshold)
    metrics.update(
        {
            "baseline_auc": metrics["roc_auc"],
            "threshold": threshold,
            "n_train": len(y_tr),
            "n_holdout": len(y_te),
            "baseline_default_rate": float(np.mean(np.r_[y_tr, y_te])),
            "algorithm": ALGORITHM,
            "evaluation": "25% stratified holdout of the baseline",
            "version": 1,
        }
    )
    metrics["_holdout"] = (hold_prob, y_te)
    return model, metrics


def save_model(
    model: Pipeline,
    dataset_id: int,
    version: int = 1,
    holdout: tuple[np.ndarray, np.ndarray] | None = None,
) -> str:
    """Persist the model artifact (and its holdout signal) and return its path.

    Version 1 keeps the original file name so existing artifacts stay valid.
    The holdout predicted probabilities and labels are saved next to the model
    (``.holdout.npz``): they are the no-drift reference segment for ADWIN.
    """
    directory = Path(settings.artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "" if version == 1 else f"_v{version}"
    path = directory / f"banking_model_ds{dataset_id}{suffix}.joblib"
    joblib.dump(model, path)
    if holdout is not None:
        np.savez(_holdout_path(path), prob=holdout[0], y=holdout[1])
    return str(path)


def _holdout_path(model_path: str | Path) -> Path:
    return Path(str(model_path) + ".holdout.npz")


def load_holdout(model_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Holdout (probabilities, labels) saved with a model, if present."""
    path = _holdout_path(model_path)
    if not path.exists():
        return None
    data = np.load(path)
    return data["prob"], data["y"]


def load_model(path: str) -> Pipeline:
    """Load a persisted model artifact."""
    return joblib.load(path)


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Predicted default probability for each record."""
    return model.predict_proba(_features(df))[:, 1]


def shap_importance(
    model: Pipeline, df: pd.DataFrame, seed: int = settings.random_seed
) -> dict[str, float]:
    """Mean |SHAP| per original feature on a batch sample, scaled to [0, 1]."""
    x = _features(df)
    if len(x) > settings.shap_sample_size:
        x = x.sample(settings.shap_sample_size, random_state=seed)
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    xt = pre.transform(x)
    values = shap.TreeExplainer(clf).shap_values(xt)
    if isinstance(values, list):  # older SHAP: one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:  # trailing class dimension
        values = values[..., -1]
    mean_abs = np.abs(values).mean(axis=0)
    names = pre.get_feature_names_out()
    raw = dict.fromkeys(MONITORED_FEATURES, 0.0)
    for name, val in zip(names, mean_abs, strict=True):
        block, _, rest = name.partition("__")
        source = rest
        if block == "cat":
            source = next(c for c in CATEGORICAL_FEATURES if rest.startswith(c))
        raw[source] += float(val)
    top = max(raw.values()) or 1.0
    return {k: v / top for k, v in raw.items()}


def permutation_sensitivity(
    model: Pipeline,
    df: pd.DataFrame,
    repeats: int = settings.sensitivity_repeats,
    seed: int = settings.random_seed,
) -> dict[str, float]:
    """Mean |Δ predicted probability| when one feature is permuted within the batch."""
    rng = np.random.default_rng(seed)
    x = _features(df).reset_index(drop=True)
    base = model.predict_proba(x)[:, 1]
    raw = {}
    for col in MONITORED_FEATURES:
        deltas = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            deltas.append(np.mean(np.abs(model.predict_proba(shuffled)[:, 1] - base)))
        raw[col] = float(np.mean(deltas))
    top = max(raw.values()) or 1.0
    return {k: v / top for k, v in raw.items()}


def performance_state(
    y: np.ndarray,
    prob: np.ndarray,
    baseline_auc: float | None,
    b: int = settings.perf_bootstrap_b,
    seed: int = settings.random_seed,
) -> dict[str, Any]:
    """Bootstrap AUC CI and the SPEC 12 state (stable/degraded/severe)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], prob[idx]))
    if not aucs or baseline_auc is None:
        return {"state": "unknown", "auc_ci": None, "auc_drop": None}
    lo, hi = (float(v) for v in np.quantile(aucs, [0.025, 0.975]))
    auc = float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")
    if hi < baseline_auc - settings.perf_severe_drop:
        state = "severe"
    elif hi < baseline_auc - settings.perf_meaningful_drop:
        state = "degraded"
    else:
        state = "stable"
    return {
        "state": state,
        "auc_ci": [lo, hi],
        "auc_drop": float(baseline_auc - auc),
    }


def probability_histogram(prob: np.ndarray, bins: int = 20) -> dict[str, list]:
    """Histogram of predicted default probabilities on fixed [0, 1] bins."""
    counts, edges = np.histogram(prob, bins=bins, range=(0, 1))
    total = max(int(counts.sum()), 1)
    return {"edges": edges.tolist(), "proportions": (counts / total).tolist()}
