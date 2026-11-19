"""Banking drift analysis: feature drift, DIS, model health and risk decision.

Pipeline for one comparison (baseline vs. a production population):

1. PSI, KS (numerical) and JSD per feature with the existing detectors.
2. Null-floor normalisation (bootstrap noise floor, SPEC 10) -> ``D_i``.
3. FI and S from the baseline credit-risk model (SPEC 11.1/11.2).
4. ``DIS_i = D_i * FI_i * S_i`` and ablations via ``dis_variants`` (SPEC 11.3).
5. Batch drift level from ``max_i DIS_i`` (SPEC 11.4).
6. Performance state from labels when available (SPEC 12).
7. ``decide`` (SPEC 13) -> alert, mapped to a banking risk category.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.services.datasets import (
    FEATURE_LABELS,
    LABEL_COLUMN,
    MONITORED_FEATURES,
    NUMERICAL_FEATURES,
)
from app.services.drift.calibration import noise_floors
from app.services.drift.jsd import jsd_from_counts
from app.services.drift.ks import ks
from app.services.drift.normalise import aggregate_d, normalise
from app.services.drift.psi import binned_proportions, categorical_psi, psi
from app.services.explain.dis import dis_variants
from app.services.monitoring import model as credit_model
from app.services.recommendations.risk import risk_assessment

logger = logging.getLogger(__name__)

MIN_FEATURE_ROWS = 30
CHART_BINS = 30

DRIFT_STATUS = {
    "low": "Stable",
    "moderate": "Moderate Drift",
    "high": "Significant Drift",
}


def severity_of(d: float) -> str:
    """Per-feature severity from normalised drift ``D_i`` (SPEC 10.3)."""
    if d >= settings.severity_high:
        return "high"
    if d >= settings.severity_moderate:
        return "moderate"
    return "low"


def batch_drift_level(overall_dis: float) -> str:
    """Batch drift level from ``max_i DIS_i`` (SPEC 11.4)."""
    if overall_dis >= settings.batch_dis_high:
        return "high"
    if overall_dis >= settings.batch_dis_moderate:
        return "moderate"
    return "low"


def _numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _categorical(series: pd.Series) -> np.ndarray:
    return series.dropna().astype(str).str.strip().str.lower().to_numpy(dtype=object)


def _stats(values: np.ndarray, total: int) -> dict[str, float | None]:
    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
            "missing_rate": 1.0 if total else None,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "missing_rate": float(1 - values.size / total) if total else 0.0,
    }


def numeric_distribution(
    arrays: dict[str, np.ndarray], bins: int = CHART_BINS
) -> dict[str, Any]:
    """Shared-edge histograms (proportions) for overlaid charts.

    Edges span the pooled 0.5–99.5 percentiles; values beyond are counted in
    the outer bins so no mass is dropped.
    """
    pooled = np.concatenate([a for a in arrays.values() if a.size]) if arrays else []
    if len(pooled) == 0:
        return {"kind": "numerical", "edges": [], "series": {}}
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    if np.all(np.mod(pooled, 1) == 0):
        # Discrete (integer) attributes: integer-aligned bins avoid aliasing.
        lo, hi = float(np.floor(lo)), float(np.ceil(hi))
        step = max(1.0, float(np.ceil((hi - lo + 1) / bins)))
        edges = np.arange(lo - 0.5, hi + 0.5 + step, step)
        lo, hi = float(edges[0]), float(edges[-1])
    else:
        if hi <= lo:
            lo, hi = float(lo) - 0.5, float(hi) + 0.5
        edges = np.linspace(lo, hi, bins + 1)
    series = {}
    for name, values in arrays.items():
        counts = np.histogram(np.clip(values, lo, hi), bins=edges)[0]
        series[name] = (counts / max(int(counts.sum()), 1)).tolist()
    return {"kind": "numerical", "edges": edges.tolist(), "series": series}


def categorical_distribution(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Category proportions over the union of observed categories."""
    cats = sorted({c for a in arrays.values() for c in a.tolist()})
    series = {}
    for name, values in arrays.items():
        total = max(len(values), 1)
        series[name] = [float(np.sum(values == c) / total) for c in cats]
    return {"kind": "categorical", "categories": cats, "series": series}


def compute_feature_drift(
    ref_series: pd.Series,
    prod_series: pd.Series,
    feature_name: str,
    feature_type: str = "numerical",
    floors: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """PSI, KS, JSD, normalised D and severity for one feature.

    Missing and infinite values are excluded per feature. Features with fewer
    than ``MIN_FEATURE_ROWS`` usable values on either side are reported as
    ``Insufficient data`` with null metrics and ``D = 0`` (never fabricated).
    """
    numerical = feature_type == "numerical"
    ref = _numeric(ref_series) if numerical else _categorical(ref_series)
    prod = _numeric(prod_series) if numerical else _categorical(prod_series)
    base = {
        "feature": feature_name,
        "label": FEATURE_LABELS.get(feature_name, feature_name),
        "type": feature_type,
        "n_reference": int(ref.size),
        "n_production": int(prod.size),
        "reference_stats": _stats(ref, len(ref_series)) if numerical else None,
        "production_stats": _stats(prod, len(prod_series)) if numerical else None,
    }
    if ref.size < MIN_FEATURE_ROWS or prod.size < MIN_FEATURE_ROWS:
        return {
            **base,
            "psi": None,
            "ks_stat": None,
            "ks_p": None,
            "jsd": None,
            "floors": None,
            "normalised": None,
            "d": 0.0,
            "severity": "low",
            "drift_status": "Insufficient data",
            "conventional_alert": False,
        }
    floors = floors or {"psi": 0.05, "ks": 0.05, "jsd": 0.02}

    if numerical:
        val_psi = psi(ref, prod)
        ks_stat, ks_p = ks(ref, prod)
        p, q, _ = binned_proportions(ref, prod)
        val_jsd = jsd_from_counts(q, p)
    else:
        val_psi = categorical_psi(ref, prod)
        ks_stat, ks_p = None, None
        cats = np.unique(np.concatenate([ref, prod]))
        val_jsd = jsd_from_counts(
            np.array([np.sum(ref == c) for c in cats]),
            np.array([np.sum(prod == c) for c in cats]),
        )

    normalised = {
        "psi": normalise(val_psi, floors["psi"], settings.norm_cap_psi),
        "jsd": normalise(val_jsd, floors["jsd"], settings.norm_cap_jsd),
    }
    if numerical:
        normalised["ks"] = normalise(ks_stat, floors["ks"], settings.norm_cap_ks)
    d = aggregate_d(list(normalised.values()))
    severity = severity_of(d)
    conventional = bool(
        val_psi > settings.conventional_psi
        or (ks_p is not None and ks_p < settings.conventional_ks_p)
        or val_jsd > settings.conventional_jsd
    )
    if not numerical:
        r_cats, p_cats = set(ref.tolist()), set(prod.tolist())
        base["new_categories"] = sorted(p_cats - r_cats)
        base["missing_categories"] = sorted(r_cats - p_cats)
    return {
        **base,
        "psi": float(val_psi),
        "ks_stat": None if ks_stat is None else float(ks_stat),
        "ks_p": None if ks_p is None else float(ks_p),
        "jsd": float(val_jsd),
        "floors": floors,
        "normalised": normalised,
        "d": float(d),
        "severity": severity,
        "drift_status": DRIFT_STATUS[severity],
        "conventional_alert": conventional,
    }


def evaluate_model_health(
    model: Any | None,
    model_metrics: dict[str, Any] | None,
    prod_df: pd.DataFrame,
) -> dict[str, Any]:
    """Score the batch and, when labels exist, compare with baseline metrics."""
    if model is None or not model_metrics:
        return {
            "has_model": False,
            "has_production_labels": False,
            "performance_state": "unknown",
            "status_message": (
                "No baseline model: the registered baseline has no usable loan_default "
                "labels, so performance cannot be evaluated and DIS uses drift only."
            ),
            "baseline_metrics": None,
            "production_metrics": None,
        }
    prob = credit_model.predict_proba(model, prod_df)
    threshold = model_metrics["threshold"]
    health: dict[str, Any] = {
        "has_model": True,
        "baseline_metrics": {
            k: model_metrics.get(k)
            for k in (
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "confusion_matrix",
                "actual_default_rate",
                "predicted_default_rate",
                "threshold",
                "n_holdout",
                "n_train",
                "baseline_default_rate",
            )
        },
        "predicted_default_rate": float(np.mean(prob >= threshold)),
        "mean_predicted_probability": float(np.mean(prob)),
        "probability_histogram": credit_model.probability_histogram(prob),
    }
    labels = (
        pd.to_numeric(prod_df[LABEL_COLUMN], errors="coerce")
        if LABEL_COLUMN in prod_df.columns
        else pd.Series(np.nan, index=prod_df.index)
    )
    mask = labels.notna().to_numpy()
    n_labels = int(mask.sum())
    health["n_labels"] = n_labels
    if n_labels < settings.min_labels:
        health.update(
            {
                "has_production_labels": False,
                "performance_state": "unknown",
                "production_metrics": None,
                "status_message": (
                    f"Production outcome labels pending ({n_labels} of the required "
                    f"{settings.min_labels}). Statistical drift does not by itself mean "
                    "accuracy has decreased; model performance degradation cannot yet "
                    "be confirmed."
                ),
            }
        )
        return health
    y = labels[mask].astype(int).to_numpy()
    metrics = credit_model.classification_metrics(y, prob[mask], threshold)
    perf = credit_model.performance_state(
        y, prob[mask], model_metrics.get("baseline_auc")
    )
    metrics.update(perf)
    messages = {
        "stable": "Labelled outcomes show no meaningful AUC drop versus the baseline.",
        "degraded": "Confirmed degradation: production AUC is meaningfully below baseline.",
        "severe": "Confirmed severe degradation: production AUC is far below baseline.",
        "unknown": "Labels present but AUC could not be estimated (single outcome class).",
    }
    health.update(
        {
            "has_production_labels": True,
            "performance_state": perf["state"],
            "production_metrics": metrics,
            "status_message": messages[perf["state"]],
        }
    )
    return health


def analyze_banking_drift(
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    batch_name: str = "Production Batch",
    model: Any | None = None,
    model_metrics: dict[str, Any] | None = None,
    floor_key: str = "adhoc",
) -> dict[str, Any]:
    """End-to-end comparison of a production population against the baseline."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    feature_results: list[dict[str, Any]] = []
    distributions: dict[str, Any] = {}
    n_prod = len(prod_df)

    for feat in MONITORED_FEATURES:
        if feat not in ref_df.columns or feat not in prod_df.columns:
            continue
        kind = "numerical" if feat in NUMERICAL_FEATURES else "categorical"
        ref_vals = (
            _numeric(ref_df[feat])
            if kind == "numerical"
            else _categorical(ref_df[feat])
        )
        prod_vals = (
            _numeric(prod_df[feat])
            if kind == "numerical"
            else _categorical(prod_df[feat])
        )
        floors = None
        if ref_vals.size >= MIN_FEATURE_ROWS and prod_vals.size >= MIN_FEATURE_ROWS:
            floors = noise_floors(f"{floor_key}:{feat}", ref_vals, prod_vals.size, kind)
        feature_results.append(
            compute_feature_drift(ref_df[feat], prod_df[feat], feat, kind, floors)
        )
        arrays = {"baseline": ref_vals, "production": prod_vals}
        distributions[feat] = (
            numeric_distribution(arrays)
            if kind == "numerical"
            else categorical_distribution(arrays)
        )
    timings["drift_ms"] = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    names = [f["feature"] for f in feature_results]
    d = {f["feature"]: min(1.0, max(0.0, f["d"])) for f in feature_results}
    if model is not None:
        fi_all = credit_model.shap_importance(model, prod_df)
        s_all = credit_model.permutation_sensitivity(model, prod_df)
        fi = {k: fi_all[k] for k in names}
        s = {k: s_all[k] for k in names}
        dis_mode = "model"
    else:
        fi = dict.fromkeys(names, 1.0)
        s = dict.fromkeys(names, 1.0)
        dis_mode = "drift_only"
    variants = dis_variants(d, fi, s) if names else {}
    for fr in feature_results:
        v = variants[fr["feature"]]
        fr.update(
            {
                "fi": fi[fr["feature"]],
                "s": s[fr["feature"]],
                "dis": v["dis"],
                "variants": v,
            }
        )
    timings["explain_ms"] = (time.perf_counter() - t1) * 1000

    overall_dis = max((f["dis"] for f in feature_results), default=0.0)
    ranked = sorted(feature_results, key=lambda f: f["dis"], reverse=True)
    for rank, fr in enumerate(ranked, start=1):
        fr["dis_rank"] = rank
    drift_level = batch_drift_level(overall_dis)

    t2 = time.perf_counter()
    health = evaluate_model_health(model, model_metrics, prod_df)
    timings["model_health_ms"] = (time.perf_counter() - t2) * 1000
    perf = health["performance_state"]
    assessment = risk_assessment(
        drift_level, perf, feature_results, health.get("production_metrics")
    )

    def _rate(df: pd.DataFrame) -> float | None:
        if LABEL_COLUMN not in df.columns:
            return None
        lab = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").dropna()
        return float(lab.mean()) if len(lab) else None

    timings["total_ms"] = (time.perf_counter() - t0) * 1000
    return {
        "batch_name": batch_name,
        "customer_count": int(n_prod),
        "baseline_count": len(ref_df),
        "features_monitored": len(feature_results),
        "drifting_features_count": sum(
            1 for f in feature_results if f["severity"] != "low"
        ),
        "significant_drift_count": sum(
            1 for f in feature_results if f["severity"] == "high"
        ),
        "conventional_alert_count": sum(
            1 for f in feature_results if f["conventional_alert"]
        ),
        "overall_dis": float(overall_dis),
        "top_drifting_feature": ranked[0]["feature"] if ranked else None,
        "top_features": [f["feature"] for f in ranked[:3]],
        "drift_level": drift_level,
        "performance_state": perf,
        "dis_mode": dis_mode,
        **assessment,
        "feature_details": feature_results,
        "distributions": distributions,
        "model_health": health,
        "default_rates": {
            "baseline_actual": _rate(ref_df),
            "production_actual": (
                _rate(prod_df) if health.get("has_production_labels") else None
            ),
            "production_predicted": health.get("predicted_default_rate"),
        },
        "thresholds": {
            "severity_moderate": settings.severity_moderate,
            "severity_high": settings.severity_high,
            "batch_dis_moderate": settings.batch_dis_moderate,
            "batch_dis_high": settings.batch_dis_high,
            "null_quantile": settings.null_quantile,
            "min_labels": settings.min_labels,
        },
        "timings": timings,
    }
