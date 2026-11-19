"""Persistence for the banking platform (PostgreSQL + MongoDB).

PostgreSQL (relational integrity, decisions):
    datasets, features (reference profile in ``reference_stats``), models
    (champion + metrics), monitoring_jobs, drift_summaries, alerts.
MongoDB (flexible / large documents):
    banking_records      one document per customer record (``owner`` key)
    baseline_registry    baseline registration metadata
    production_batches   batch metadata and processing history
    root_cause_reports   full analysis report per monitoring job
    drift_events         per-feature detector values per job
    monitoring_logs      audit trail
"""

import hashlib
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import mongo
from app.database.models import (
    Alert,
    Dataset,
    DriftSummary,
    Feature,
    Model,
    MonitoringJob,
    User,
)
from app.services.datasets import (
    BANKING_OPTIONAL_COLUMNS,
    BANKING_REQUIRED_COLUMNS,
    CATEGORICAL_FEATURES,
    LABEL_COLUMN,
    MONITORED_FEATURES,
    NUMERICAL_FEATURES,
    json_records,
)
from app.services.monitoring import model as credit_model

logger = logging.getLogger(__name__)

BASELINE_SOURCE = "banking-baseline"
STORED_COLUMNS = BANKING_REQUIRED_COLUMNS + BANKING_OPTIONAL_COLUMNS


def now() -> datetime:
    """Naive UTC timestamp (matches the TIMESTAMP columns)."""
    return datetime.now(UTC).replace(tzinfo=None)


def iso(ts: datetime | None) -> str | None:
    """ISO-8601 UTC string."""
    return None if ts is None else ts.replace(microsecond=0).isoformat() + "Z"


def db():
    """Return the Mongo database (indirection keeps tests patchable)."""
    return mongo.database()


def content_hash(df: pd.DataFrame) -> str:
    """Stable SHA-256 of the cleaned dataset content."""
    payload = df[STORED_COLUMNS].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def audit(
    event: str,
    user: User | None,
    details: dict[str, Any] | None = None,
    level: str = "info",
) -> None:
    """Append an audit event; failures are logged, never raised."""
    try:
        db().monitoring_logs.insert_one(
            {
                "timestamp": now(),
                "event": event,
                "level": level,
                "user_id": user.user_id if user else None,
                "user_email": user.email if user else None,
                "user_role": user.role if user else None,
                "details": details or {},
            }
        )
    except Exception:
        logger.exception("Failed to write audit event %s", event)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _owner_baseline(dataset_id: int) -> str:
    return f"baseline:{dataset_id}"


def _owner_batch(batch_id: str) -> str:
    return f"batch:{batch_id}"


def write_records(owner: str, df: pd.DataFrame) -> None:
    """Replace the records stored for ``owner``."""
    coll = db().banking_records
    coll.delete_many({"owner": owner})
    rows = json_records(df[STORED_COLUMNS])
    for start in range(0, len(rows), 5000):
        chunk = [{"owner": owner, **r} for r in rows[start : start + 5000]]
        if chunk:
            coll.insert_many(chunk, ordered=False)


def read_records(owner: str) -> pd.DataFrame:
    """Load stored records for ``owner`` as a typed DataFrame."""
    docs = list(db().banking_records.find({"owner": owner}, {"_id": 0, "owner": 0}))
    df = pd.DataFrame(docs, columns=STORED_COLUMNS)
    for col in NUMERICAL_FEATURES + [LABEL_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def reference_profile(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per-feature reference statistics stored in ``features.reference_stats``."""
    profile = {}
    for col in NUMERICAL_FEATURES:
        v = df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        profile[col] = {
            "count": int(v.size),
            "missing_rate": float(1 - v.size / max(len(df), 1)),
            "mean": float(np.mean(v)) if v.size else None,
            "std": float(np.std(v)) if v.size else None,
            "quantiles": (
                {
                    str(q): float(np.quantile(v, q))
                    for q in (0.05, 0.25, 0.5, 0.75, 0.95)
                }
                if v.size
                else {}
            ),
            "min": float(np.min(v)) if v.size else None,
            "max": float(np.max(v)) if v.size else None,
        }
    for col in CATEGORICAL_FEATURES:
        counts = df[col].dropna().astype(str).value_counts(normalize=True)
        profile[col] = {
            "count": int(df[col].notna().sum()),
            "missing_rate": float(df[col].isna().mean()),
            "proportions": {k: float(v) for k, v in counts.items()},
        }
    return profile


def find_baseline_by_hash(sha: str) -> dict[str, Any] | None:
    """Return baseline metadata with identical content, if any."""
    return db().baseline_registry.find_one({"content_sha256": sha}, {"_id": 0})


def _unique_dataset_name(session: Session, user_id: int, name: str) -> str:
    existing = set(
        session.scalars(select(Dataset.name).where(Dataset.user_id == user_id))
    )
    if name not in existing:
        return name
    version = 2
    while f"{name} (v{version})" in existing:
        version += 1
    return f"{name} (v{version})"


def register_baseline(
    session: Session,
    user: User,
    df: pd.DataFrame,
    name: str,
    filename: str | None,
    synthetic_columns: list[str],
    is_synthetic: bool,
) -> dict[str, Any]:
    """Register the baseline: dataset, features, champion model, records.

    PostgreSQL rows are committed only after the model trains and the Mongo
    writes succeed; on failure the transaction rolls back and Mongo writes
    are compensated.
    """
    sha = content_hash(df)
    name = _unique_dataset_name(session, user.user_id, name)
    dataset = Dataset(
        user_id=user.user_id,
        name=name,
        source=f"{BASELINE_SOURCE}:{(filename or 'upload')[:150]}",
        row_count=len(df),
    )
    session.add(dataset)
    session.flush()
    profile = reference_profile(df)
    for col in MONITORED_FEATURES:
        session.add(
            Feature(
                dataset_id=dataset.dataset_id,
                name=col,
                type="numerical" if col in NUMERICAL_FEATURES else "categorical",
                reference_stats=profile[col],
            )
        )
    try:
        model_obj, metrics = credit_model.train_baseline_model(df)
        metrics["artifact_path"] = credit_model.save_model(
            model_obj, dataset.dataset_id
        )
        metrics["has_model"] = True
        algorithm = credit_model.ALGORITHM
    except credit_model.ModelUnavailableError as exc:
        metrics = {"has_model": False, "reason": str(exc)}
        algorithm = "none-drift-only"
    model_row = Model(
        dataset_id=dataset.dataset_id,
        algorithm=algorithm,
        version=1,
        status="champion",
        metrics=metrics,
    )
    session.add(model_row)
    session.flush()
    owner = _owner_baseline(dataset.dataset_id)
    meta = {
        "dataset_id": dataset.dataset_id,
        "model_id": model_row.model_id,
        "name": name,
        "filename": filename,
        "row_count": len(df),
        "labelled_rows": int(df[LABEL_COLUMN].notna().sum()),
        "content_sha256": sha,
        "is_synthetic": bool(is_synthetic),
        "synthetic_columns": synthetic_columns,
        "created_at": now(),
        "created_by": user.email,
    }
    try:
        write_records(owner, df)
        db().baseline_registry.insert_one(dict(meta))
        session.commit()
    except Exception:
        session.rollback()
        db().banking_records.delete_many({"owner": owner})
        db().baseline_registry.delete_one({"dataset_id": meta["dataset_id"]})
        raise
    return baseline_summary(session, meta)


def current_baseline_meta() -> dict[str, Any] | None:
    """Most recently registered baseline (the active reference population)."""
    return db().baseline_registry.find_one(
        {}, {"_id": 0}, sort=[("created_at", DESCENDING)]
    )


def baseline_summary(session: Session, meta: dict[str, Any]) -> dict[str, Any]:
    """Baseline metadata joined with its PostgreSQL dataset/model/features."""
    model_row = session.get(Model, meta["model_id"])
    features = session.scalars(
        select(Feature)
        .where(Feature.dataset_id == meta["dataset_id"])
        .order_by(Feature.feature_id)
    ).all()
    metrics = dict(model_row.metrics or {}) if model_row else {}
    metrics.pop("artifact_path", None)
    return {
        "has_baseline": True,
        "dataset_id": meta["dataset_id"],
        "model_id": meta["model_id"],
        "dataset_name": meta["name"],
        "filename": meta.get("filename"),
        "row_count": meta["row_count"],
        "labelled_rows": meta.get("labelled_rows", 0),
        "is_synthetic": meta.get("is_synthetic", False),
        "synthetic_columns": meta.get("synthetic_columns", []),
        "created_at": iso(meta.get("created_at")),
        "created_by": meta.get("created_by"),
        "model": {
            "algorithm": model_row.algorithm if model_row else None,
            "status": model_row.status if model_row else None,
            "trained_at": iso(model_row.trained_at) if model_row else None,
            "metrics": metrics,
        },
        "features": [
            {"name": f.name, "type": f.type, "reference_stats": f.reference_stats}
            for f in features
        ],
    }


def load_baseline() -> tuple[dict[str, Any], pd.DataFrame]:
    """Current baseline metadata and records; raises ``LookupError`` if none."""
    meta = current_baseline_meta()
    if not meta:
        raise LookupError("No historical baseline is registered.")
    return meta, read_records(_owner_baseline(meta["dataset_id"]))


@lru_cache(maxsize=8)
def _load_model_cached(path: str):
    return credit_model.load_model(path)


def load_champion(
    session: Session, model_id: int
) -> tuple[Any | None, dict[str, Any] | None]:
    """Return (model, metrics) or (None, None) when the baseline has no model."""
    row = session.get(Model, model_id)
    if not row or not (row.metrics or {}).get("has_model"):
        return None, None
    return _load_model_cached(row.metrics["artifact_path"]), row.metrics


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def get_batch(batch_id: str) -> dict[str, Any] | None:
    """Batch metadata (without records)."""
    return db().production_batches.find_one({"batch_id": batch_id}, {"_id": 0})


def find_batch_by_hash(sha: str) -> dict[str, Any] | None:
    """Batch with identical content, if any."""
    return db().production_batches.find_one({"content_sha256": sha}, {"_id": 0})


def save_batch(
    user: User,
    batch_id: str,
    df: pd.DataFrame,
    filename: str | None,
    batch_date: str,
    description: str,
    is_synthetic: bool,
    synthetic_columns: list[str],
    scenario: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Store batch records and metadata; keeps processing history on overwrite."""
    previous = get_batch(batch_id)
    labelled = int(df[LABEL_COLUMN].notna().sum())
    meta = {
        "batch_id": batch_id,
        "filename": filename,
        "batch_date": batch_date,
        "description": description,
        "row_count": len(df),
        "labelled_rows": labelled,
        "has_labels": labelled > 0,
        "content_sha256": content_hash(df),
        "is_synthetic": bool(is_synthetic),
        "synthetic_columns": synthetic_columns,
        "scenario": scenario,
        "uploaded_at": now(),
        "uploaded_by": user.email,
        "status": "registered",
        "latest_job_id": None,
        "processing_history": (previous or {}).get("processing_history", []),
        "upload_count": (previous or {}).get("upload_count", 0) + 1,
    }
    write_records(_owner_batch(batch_id), df)
    db().production_batches.replace_one({"batch_id": batch_id}, meta, upsert=True)
    if previous and overwrite:
        meta["overwritten"] = True
    return meta


def load_batch_records(batch_id: str) -> pd.DataFrame:
    """Records of a stored batch."""
    return read_records(_owner_batch(batch_id))


def list_batches() -> list[dict[str, Any]]:
    """All batches ordered by batch date, with the latest analysis summary."""
    batches = list(
        db()
        .production_batches.find({}, {"_id": 0})
        .sort([("batch_date", ASCENDING), ("batch_id", ASCENDING)])
    )
    job_ids = [b["latest_job_id"] for b in batches if b.get("latest_job_id")]
    reports = {
        r["job_id"]: r
        for r in db().root_cause_reports.find(
            {"job_id": {"$in": job_ids}}, SUMMARY_PROJECTION
        )
    }
    out = []
    for b in batches:
        b["uploaded_at"] = iso(b.get("uploaded_at"))
        for h in b.get("processing_history", []):
            h["at"] = iso(h["at"]) if isinstance(h.get("at"), datetime) else h.get("at")
        b["latest_analysis"] = _summary(reports.get(b.get("latest_job_id")))
        out.append(b)
    return out


# ---------------------------------------------------------------------------
# Monitoring jobs and reports
# ---------------------------------------------------------------------------

SUMMARY_FIELDS = (
    "job_id",
    "analysis_type",
    "batch_id",
    "batch_ids",
    "batch_date",
    "batch_name",
    "customer_count",
    "overall_dis",
    "drift_level",
    "risk_category",
    "alert_type",
    "performance_state",
    "drifting_features_count",
    "significant_drift_count",
    "top_drifting_feature",
    "top_features",
    "analysed_at",
    "analysed_by",
    "baseline_dataset_id",
    "is_synthetic",
    "dis_mode",
    "features_monitored",
)
SUMMARY_PROJECTION = {"_id": 0, **dict.fromkeys(SUMMARY_FIELDS, 1)}


def _summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    out = {k: report.get(k) for k in SUMMARY_FIELDS}
    out["analysed_at"] = (
        iso(out["analysed_at"])
        if isinstance(out["analysed_at"], datetime)
        else out["analysed_at"]
    )
    return out


def _parse_date(value: str | None) -> datetime:
    try:
        return datetime.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return now()


def start_job(
    session: Session, user: User, model_id: int, job_batch_id: str, dates: list[str]
) -> MonitoringJob:
    """Insert and commit a ``running`` monitoring job."""
    parsed = sorted(_parse_date(d) for d in dates) or [now()]
    job = MonitoringJob(
        model_id=model_id,
        user_id=user.user_id,
        batch_id=job_batch_id[:64],
        window_start=parsed[0],
        window_end=parsed[-1],
        status="running",
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


FLOAT4_MIN = 1.2e-38
FLOAT4_MAX = 3.4e38


def to_real(value: float | None) -> float | None:
    """Clamp to PostgreSQL REAL (float4) range.

    ``drift_summaries`` uses REAL columns; e.g. a KS p-value of 1e-300 would
    underflow. Sub-normal magnitudes become 0.0 (full precision is kept in the
    MongoDB report).
    """
    if value is None or not np.isfinite(value):
        return None
    if abs(value) < FLOAT4_MIN:
        return 0.0
    return float(max(-FLOAT4_MAX, min(FLOAT4_MAX, value)))


def complete_job(
    session: Session,
    job: MonitoringJob,
    baseline_dataset_id: int,
    report: dict[str, Any],
) -> None:
    """Write drift_summaries, alert, Mongo report/events; commit atomically.

    The Mongo report is written before the PostgreSQL commit and removed again
    if the commit fails, so the two stores never disagree about a completed job.
    """
    features = {
        f.name: f.feature_id
        for f in session.scalars(
            select(Feature).where(Feature.dataset_id == baseline_dataset_id)
        )
    }
    for fr in report["feature_details"]:
        if fr["feature"] not in features:
            continue
        session.add(
            DriftSummary(
                job_id=job.job_id,
                feature_id=features[fr["feature"]],
                psi=to_real(fr["psi"]),
                ks_stat=to_real(fr["ks_stat"]),
                ks_p=to_real(fr["ks_p"]),
                jsd=to_real(fr["jsd"]),
                dis=to_real(fr["dis"]),
                severity=fr["severity"],
            )
        )
    alert_id = None
    if report["alert_type"]:
        duplicate = session.scalar(
            select(Alert).where(
                Alert.job_id == job.job_id, Alert.alert_type == report["alert_type"]
            )
        )
        if not duplicate:
            evidence = "; ".join(report["evidence"])
            alert = Alert(
                job_id=job.job_id,
                alert_type=report["alert_type"],
                level=report["alert_level"],
                recommendation=f"[{report['risk_category']}] {report['recommendation']}. Evidence: {evidence}",
            )
            session.add(alert)
            session.flush()
            alert_id = alert.alert_id
    report["alert_id"] = alert_id
    job.status = "completed"
    m = db()
    m.root_cause_reports.replace_one({"job_id": job.job_id}, report, upsert=True)
    m.drift_events.insert_many(
        [
            {
                "job_id": job.job_id,
                "batch_id": report["batch_id"],
                "feature": fr["feature"],
                "psi": fr["psi"],
                "ks_stat": fr["ks_stat"],
                "ks_p": fr["ks_p"],
                "jsd": fr["jsd"],
                "floors": fr["floors"],
                "normalised": fr["normalised"],
                "d": fr["d"],
                "dis": fr["dis"],
                "severity": fr["severity"],
                "exceeded_conventional": fr["conventional_alert"],
                "histogram": report["distributions"].get(fr["feature"]),
                "created_at": report["analysed_at"],
            }
            for fr in report["feature_details"]
        ]
    )
    try:
        session.commit()
    except Exception:
        session.rollback()
        m.root_cause_reports.delete_one({"job_id": job.job_id})
        m.drift_events.delete_many({"job_id": job.job_id})
        raise


def fail_job(session: Session, job: MonitoringJob, error: str) -> None:
    """Mark a job failed and record the error."""
    session.rollback()
    job = session.get(MonitoringJob, job.job_id)
    if job:
        job.status = "failed"
        session.commit()
    try:
        db().monitoring_logs.insert_one(
            {
                "timestamp": now(),
                "event": "analysis_failed",
                "level": "error",
                "details": {"job_id": job.job_id if job else None, "error": error},
            }
        )
    except Exception:
        logger.exception("Failed to log job failure")


def record_batch_processing(
    batch_ids: list[str], job_id: int, user: User, analysis_type: str
) -> None:
    """Append to processing history; individual runs also update ``latest_job_id``."""
    entry = {"job_id": job_id, "at": now(), "by": user.email, "type": analysis_type}
    update: dict[str, Any] = {"$push": {"processing_history": entry}}
    if analysis_type == "individual":
        update["$set"] = {"latest_job_id": job_id, "status": "analysed"}
    db().production_batches.update_many({"batch_id": {"$in": batch_ids}}, update)


def get_report(job_id: int) -> dict[str, Any] | None:
    """Full stored report for a job."""
    return db().root_cause_reports.find_one({"job_id": job_id}, {"_id": 0})


def report_history(analysis_type: str | None = None) -> list[dict[str, Any]]:
    """Summaries of all completed analyses ordered by batch date then time."""
    query = {"analysis_type": analysis_type} if analysis_type else {}
    rows = (
        db()
        .root_cause_reports.find(query, SUMMARY_PROJECTION)
        .sort([("batch_date", ASCENDING), ("analysed_at", ASCENDING)])
    )
    return [_summary(r) for r in rows]


def job_rows(session: Session, limit: int = 200) -> list[dict[str, Any]]:
    """Monitoring jobs from PostgreSQL with user and alert (SQL join)."""
    stmt = (
        select(
            MonitoringJob.job_id,
            MonitoringJob.batch_id,
            MonitoringJob.status,
            MonitoringJob.window_start,
            MonitoringJob.window_end,
            User.email,
            Alert.alert_type,
            Alert.level,
            Alert.created_at,
            func.max(DriftSummary.dis).label("max_dis"),
            func.count(DriftSummary.summary_id).label("features"),
        )
        .join(User, User.user_id == MonitoringJob.user_id)
        .outerjoin(Alert, Alert.job_id == MonitoringJob.job_id)
        .outerjoin(DriftSummary, DriftSummary.job_id == MonitoringJob.job_id)
        .group_by(
            MonitoringJob.job_id,
            User.email,
            Alert.alert_type,
            Alert.level,
            Alert.created_at,
        )
        .order_by(MonitoringJob.job_id.desc())
        .limit(limit)
    )
    return [
        {
            "job_id": r.job_id,
            "batch_id": r.batch_id,
            "status": r.status,
            "window_start": iso(r.window_start),
            "window_end": iso(r.window_end),
            "user": r.email,
            "alert_type": r.alert_type,
            "alert_level": r.level,
            "alert_created_at": iso(r.created_at),
            "max_dis": None if r.max_dis is None else float(r.max_dis),
            "features_summarised": int(r.features),
        }
        for r in session.execute(stmt)
    ]


def audit_events(limit: int = 200, event: str | None = None) -> list[dict[str, Any]]:
    """Most recent audit events."""
    query = {"event": event} if event else {}
    rows = (
        db()
        .monitoring_logs.find(query, {"_id": 0})
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    out = []
    for r in rows:
        r["timestamp"] = (
            iso(r["timestamp"])
            if isinstance(r.get("timestamp"), datetime)
            else r.get("timestamp")
        )
        out.append(r)
    return out
