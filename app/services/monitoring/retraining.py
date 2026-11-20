"""Controlled champion-challenger retraining (human-approved, never automatic).

Workflow (each step is a separate, audited capability):

1. ``create_challenger``  train model V(n+1) on the *labelled* data only:
   baseline training split + a training share of labelled production records.
   A stratified share of the labelled production records is held out as an
   independent **post-drift test set** that is never used for fitting or for
   threshold selection. The baseline holdout is also kept aside.
2. Evaluate champion and challenger on the *same* post-drift test set (paired
   bootstrap CI of the AUC difference) and on the baseline holdout
   (no-regression guard against forgetting the original population).
3. Store V(n+1) as ``challenger`` and a ``pending`` ``retraining_events`` row;
   the full comparison is stored in MongoDB ``model_evaluations``.
4. ``deploy`` (administrator)  only if the acceptance criteria were met: one
   PostgreSQL transaction retires the champion and promotes the challenger.
5. ``reject`` / ``rollback`` restore the previous champion, keeping history.

Unlabelled production rows are never used as training data. The drift
reference population stays the registered baseline; only the model changes.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Model, RetrainingEvent, User
from app.services.datasets import LABEL_COLUMN
from app.services.monitoring import model as credit_model
from app.services.monitoring import store

logger = logging.getLogger(__name__)

POST_DRIFT_TEST_SHARE = 0.4


class RetrainingError(ValueError):
    """Raised when a retraining request cannot be satisfied safely."""


def _paired_auc_gain(
    y: np.ndarray, p_old: np.ndarray, p_new: np.ndarray, b: int, seed: int
) -> dict[str, float | None]:
    """Paired bootstrap of AUC(new) - AUC(old) on the same records."""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(b):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            gains.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_old[idx]))
    point = float(roc_auc_score(y, p_new) - roc_auc_score(y, p_old))
    if not gains:
        return {"gain": point, "ci_low": None, "ci_high": None}
    lo, hi = np.quantile(gains, [0.025, 0.975])
    return {"gain": point, "ci_low": float(lo), "ci_high": float(hi)}


def _labelled_production(batch_ids: list[str]) -> pd.DataFrame:
    frames = []
    for batch_id in batch_ids:
        if not store.get_batch(batch_id):
            raise RetrainingError(f"Batch '{batch_id}' not found.")
        df = store.load_batch_records(batch_id)
        df = df[df[LABEL_COLUMN].notna()].copy()
        df["_batch_id"] = batch_id
        frames.append(df)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(data) < settings.retrain_min_production_labels:
        raise RetrainingError(
            f"Retraining needs at least {settings.retrain_min_production_labels} "
            f"labelled production records; the selected batches have {len(data)}. "
            "Unlabelled records are never used for training."
        )
    if data[LABEL_COLUMN].nunique() < 2:
        raise RetrainingError("Labelled production records contain a single class.")
    return data


def create_challenger(
    session: Session,
    user: User,
    batch_ids: list[str],
    alert_id: int | None = None,
    seed: int = settings.random_seed,
) -> dict[str, Any]:
    """Train, evaluate and register a challenger; nothing is deployed."""
    base_meta, ref_df = store.load_baseline()
    champion_id = store.champion_model_id(session, base_meta)
    champion, champ_metrics = store.load_champion(session, champion_id)
    if champion is None:
        raise RetrainingError("The baseline has no supervised champion model.")
    pending = session.scalar(
        select(func.count(RetrainingEvent.event_id)).where(
            RetrainingEvent.old_model_id == champion_id,
            RetrainingEvent.decision == "pending",
        )
    )
    if pending:
        raise RetrainingError(
            "A challenger for the current champion is already pending review."
        )

    prod = _labelled_production(batch_ids)
    y_prod = prod[LABEL_COLUMN].astype(int).to_numpy()
    prod_tr, prod_te = train_test_split(
        prod, test_size=POST_DRIFT_TEST_SHARE, stratify=y_prod, random_state=seed
    )
    # Baseline split identical to the one used to train/evaluate V1.
    base_x_tr, base_x_te, base_y_tr, base_y_te = credit_model.split_labelled(ref_df)
    x_tr = pd.concat([base_x_tr, credit_model._features(prod_tr)], ignore_index=True)
    y_tr = np.r_[base_y_tr, prod_tr[LABEL_COLUMN].astype(int).to_numpy()]
    x_te = credit_model._features(prod_te)
    y_te = prod_te[LABEL_COLUMN].astype(int).to_numpy()

    challenger = credit_model.fit_pipeline(x_tr, y_tr, seed)
    threshold = credit_model.select_threshold(challenger, x_tr, y_tr)

    p_old = champion.predict_proba(x_te)[:, 1]
    p_new = challenger.predict_proba(x_te)[:, 1]
    gain = _paired_auc_gain(y_te, p_old, p_new, settings.perf_bootstrap_b, seed)
    old_post = credit_model.classification_metrics(y_te, p_old, champ_metrics["threshold"])
    new_post = credit_model.classification_metrics(y_te, p_new, threshold)
    old_base = credit_model.classification_metrics(
        base_y_te, champion.predict_proba(base_x_te)[:, 1], champ_metrics["threshold"]
    )
    new_base = credit_model.classification_metrics(
        base_y_te, challenger.predict_proba(base_x_te)[:, 1], threshold
    )
    no_regression = (
        new_base["roc_auc"] is not None
        and old_base["roc_auc"] is not None
        and new_base["roc_auc"] >= old_base["roc_auc"] - settings.perf_severe_drop
    )
    criteria = {
        "min_gain_auc": settings.retrain_min_gain_auc,
        "gain_meets_minimum": gain["gain"] >= settings.retrain_min_gain_auc,
        "gain_ci_excludes_zero": gain["ci_low"] is not None and gain["ci_low"] > 0,
        "no_baseline_regression": bool(no_regression),
    }
    acceptable = all(criteria[k] for k in
                     ("gain_meets_minimum", "gain_ci_excludes_zero", "no_baseline_regression"))

    version = (
        session.scalar(
            select(func.max(Model.version)).where(Model.dataset_id == base_meta["dataset_id"])
        )
        or 1
    ) + 1
    metrics = {
        **{k: new_post[k] for k in ("accuracy", "precision", "recall", "f1_score",
                                     "roc_auc", "confusion_matrix")},
        # Reference AUC for future performance monitoring of this model:
        # measured on the independent post-drift test set.
        "baseline_auc": new_post["roc_auc"],
        "threshold": threshold,
        "n_train": len(y_tr),
        "n_holdout": len(y_te),
        "baseline_default_rate": float(np.mean(y_tr)),
        "algorithm": credit_model.ALGORITHM,
        "evaluation": "independent stratified post-drift test split "
                      f"({POST_DRIFT_TEST_SHARE:.0%} of labelled production records)",
        "version": version,
        "training_batches": sorted(batch_ids),
        "parent_model_id": champion_id,
        "has_model": True,
    }
    metrics["artifact_path"] = credit_model.save_model(
        challenger, base_meta["dataset_id"], version, (p_new, y_te)
    )
    row = Model(
        dataset_id=base_meta["dataset_id"],
        algorithm=credit_model.ALGORITHM,
        version=version,
        status="challenger",
        metrics=metrics,
    )
    session.add(row)
    session.flush()
    event = RetrainingEvent(
        alert_id=alert_id, old_model_id=champion_id, new_model_id=row.model_id,
        decision="pending",
    )
    session.add(event)
    session.flush()
    evaluation = {
        "event_id": event.event_id,
        "champion_model_id": champion_id,
        "challenger_model_id": row.model_id,
        "challenger_version": version,
        "training_batches": sorted(batch_ids),
        "data": {
            "baseline_train_rows": len(base_y_tr),
            "production_train_rows": len(prod_tr),
            "post_drift_test_rows": len(y_te),
            "baseline_holdout_rows": len(base_y_te),
            "unlabelled_rows_used": 0,
        },
        "post_drift_test": {"champion": old_post, "challenger": new_post},
        "baseline_holdout": {"champion": old_base, "challenger": new_base},
        "auc_gain": gain,
        "criteria": criteria,
        "acceptable": acceptable,
        "created_at": store.now(),
        "created_by": user.email,
    }
    try:
        store.db().model_evaluations.insert_one(dict(evaluation))
        session.commit()
    except Exception:
        session.rollback()
        store.db().model_evaluations.delete_one({"event_id": evaluation["event_id"]})
        raise
    store.audit("challenger_created", user, {
        "event_id": event.event_id, "challenger_model_id": row.model_id,
        "version": version, "acceptable": acceptable, "auc_gain": gain["gain"],
    })
    return _public(evaluation)


def _public(evaluation: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in evaluation.items() if k != "_id"}
    if hasattr(out.get("created_at"), "isoformat"):
        out["created_at"] = store.iso(out["created_at"])
    return out


def _event(session: Session, event_id: int) -> RetrainingEvent:
    event = session.get(RetrainingEvent, event_id)
    if not event:
        raise LookupError(f"Retraining event {event_id} not found.")
    return event


def deploy(session: Session, user: User, event_id: int) -> dict[str, Any]:
    """Promote a validated challenger in one transaction (admin approval)."""
    event = _event(session, event_id)
    if event.decision != "pending":
        raise RetrainingError(f"Event {event_id} is '{event.decision}', not pending.")
    evaluation = store.db().model_evaluations.find_one({"event_id": event_id})
    if not evaluation or not evaluation.get("acceptable"):
        raise RetrainingError(
            "The challenger did not meet the acceptance criteria on the independent "
            "test set; unvalidated models are never deployed."
        )
    old = session.get(Model, event.old_model_id)
    new = session.get(Model, event.new_model_id)
    if old.status != "champion":
        raise RetrainingError("The challenger's parent is no longer the champion.")
    try:
        old.status = "retired"
        session.flush()  # free the one-champion-per-dataset partial unique index
        new.status = "champion"
        event.decision = "deployed"
        session.commit()
    except Exception:
        session.rollback()
        raise
    store.audit("model_deployed", user, {"event_id": event_id, "champion": new.model_id,
                                         "retired": old.model_id})
    return {"event_id": event_id, "decision": "deployed", "champion_model_id": new.model_id,
            "retired_model_id": old.model_id}


def reject(session: Session, user: User, event_id: int) -> dict[str, Any]:
    """Reject a pending challenger (it is retired, never deployed)."""
    event = _event(session, event_id)
    if event.decision != "pending":
        raise RetrainingError(f"Event {event_id} is '{event.decision}', not pending.")
    session.get(Model, event.new_model_id).status = "retired"
    event.decision = "rejected"
    session.commit()
    store.audit("challenger_rejected", user, {"event_id": event_id})
    return {"event_id": event_id, "decision": "rejected"}


def rollback(session: Session, user: User, event_id: int) -> dict[str, Any]:
    """Restore the previous champion after a deployment."""
    event = _event(session, event_id)
    if event.decision != "deployed":
        raise RetrainingError(f"Event {event_id} is '{event.decision}', not deployed.")
    old = session.get(Model, event.old_model_id)
    new = session.get(Model, event.new_model_id)
    if new.status != "champion":
        raise RetrainingError("The deployed model is no longer the champion.")
    try:
        new.status = "retired"
        session.flush()
        old.status = "champion"
        event.decision = "rolled_back"
        event.rollback_flag = True
        session.commit()
    except Exception:
        session.rollback()
        raise
    store.audit("model_rolled_back", user, {"event_id": event_id, "champion": old.model_id})
    return {"event_id": event_id, "decision": "rolled_back", "champion_model_id": old.model_id}


def list_models(session: Session) -> list[dict[str, Any]]:
    """All model versions of the active baseline dataset."""
    meta = store.current_baseline_meta()
    if not meta:
        return []
    rows = session.scalars(
        select(Model).where(Model.dataset_id == meta["dataset_id"]).order_by(Model.version)
    ).all()
    out = []
    for r in rows:
        m = dict(r.metrics or {})
        m.pop("artifact_path", None)
        out.append({"model_id": r.model_id, "version": r.version, "status": r.status,
                    "algorithm": r.algorithm, "trained_at": store.iso(r.trained_at),
                    "roc_auc": m.get("roc_auc"), "f1_score": m.get("f1_score"),
                    "evaluation": m.get("evaluation"),
                    "training_batches": m.get("training_batches", [])})
    return out


def list_events(session: Session) -> list[dict[str, Any]]:
    """Retraining events (PostgreSQL) joined with their evaluations (MongoDB)."""
    events = session.scalars(
        select(RetrainingEvent).order_by(RetrainingEvent.event_id.desc())
    ).all()
    evals = {e["event_id"]: e for e in store.db().model_evaluations.find(
        {"event_id": {"$in": [e.event_id for e in events]}}, {"_id": 0})}
    return [
        {"event_id": e.event_id, "alert_id": e.alert_id, "old_model_id": e.old_model_id,
         "new_model_id": e.new_model_id, "decision": e.decision,
         "rollback_flag": e.rollback_flag, "created_at": store.iso(e.created_at),
         "evaluation": _public(evals[e.event_id]) if e.event_id in evals else None}
        for e in events
    ]
