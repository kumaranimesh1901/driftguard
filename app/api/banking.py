"""Banking credit-risk drift monitoring API.

Roles: ``reviewer`` read-only; ``analyst`` may also upload batches, run
analyses and generate synthetic batches; ``administrator`` may additionally
register baselines and seed the demonstration scenario.
"""

import hashlib
import io
import logging
from pathlib import PurePath
from typing import Any

import numpy as np
import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import require_roles
from app.database.models import User
from app.database.postgres import get_db
from app.schemas.banking import (
    AnalysisRequest,
    GenerateBaselineRequest,
    GenerateBatchRequest,
    RunAllRequest,
    SeedDemoRequest,
)
from app.services.datasets import (
    BATCH_ID_PATTERN,
    FEATURE_LABELS,
    MONITORED_FEATURES,
    NUMERICAL_FEATURES,
    prepare_banking_frame,
)
from app.services.drift.calibration import noise_floors
from app.services.monitoring import store
from app.services.monitoring.analysis import (
    analyze_banking_drift,
    categorical_distribution,
    compute_feature_drift,
    numeric_distribution,
)
from app.services.reports import generate_committee_pdf
from app.services.synthetic_data import SCENARIOS, BankingDataGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/banking", tags=["banking"])

READ = require_roles("administrator", "analyst", "reviewer")
WRITE = require_roles("administrator", "analyst")
ADMIN = require_roles("administrator")


def _error(status: int, detail: str, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"detail": detail, "code": code})


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV with extension, size, binary and row-count checks."""
    name = PurePath(file.filename or "").name
    if not name.lower().endswith(".csv"):
        raise _error(400, f"'{name or 'file'}' is not a .csv file.", "invalid_file")
    limit = settings.max_upload_mb * 1024 * 1024
    contents = await file.read(limit + 1)
    if len(contents) > limit:
        raise _error(
            413,
            f"'{name}' exceeds the {settings.max_upload_mb} MB upload limit.",
            "file_too_large",
        )
    if not contents.strip():
        raise _error(400, f"'{name}' is empty.", "empty_file")
    if b"\x00" in contents[:4096]:
        raise _error(
            400, f"'{name}' looks like a binary file, not CSV text.", "invalid_file"
        )
    try:
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise _error(
            400, f"Could not parse '{name}' as CSV: {exc}", "invalid_file"
        ) from exc
    if len(df) > settings.max_batch_rows:
        raise _error(
            413,
            f"'{name}' has {len(df)} rows; the limit is {settings.max_batch_rows}.",
            "too_many_rows",
        )
    return df


def _batch_id_from_filename(filename: str | None) -> str:
    stem = PurePath(filename or "batch").stem
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem).strip("_-")
    return (cleaned or "batch")[:64]


_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
        start=1,
    )
}


def _infer_batch_date(filename: str | None) -> str:
    """Infer YYYY-MM-DD (month end) from names like ``March_2026.csv``; else today."""
    import calendar
    import re

    stem = PurePath(filename or "").stem.lower()
    year = re.search(r"(20\d{2})", stem)
    month = next((n for m, n in _MONTHS.items() if m in stem), None)
    if year and month:
        y = int(year.group(1))
        return f"{y:04d}-{month:02d}-{calendar.monthrange(y, month)[1]:02d}"
    return store.now().date().isoformat()


def _baseline_or_409() -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        return store.load_baseline()
    except LookupError as exc:
        raise _error(
            409,
            "Register a historical baseline dataset before analysing batches.",
            "baseline_required",
        ) from exc


def _clean_report(report: dict[str, Any]) -> dict[str, Any]:
    report.pop("_id", None)
    if hasattr(report.get("analysed_at"), "isoformat"):
        report["analysed_at"] = store.iso(report["analysed_at"])
    return report


# ---------------------------------------------------------------------------
# Datasets (baseline)
# ---------------------------------------------------------------------------


@router.post("/datasets/validate")
async def validate_dataset(
    file: UploadFile = File(...),
    fill_missing_with_synthetic: bool = Form(False),
    _: User = Depends(WRITE),
) -> dict[str, Any]:
    """Validate a CSV and return a preview without storing anything."""
    raw = await _read_csv(file)
    _, result = prepare_banking_frame(raw, fill_missing_with_synthetic)
    return {"filename": PurePath(file.filename or "").name, **result.to_dict()}


@router.post("/datasets/baseline", status_code=201)
async def register_baseline(
    file: UploadFile = File(...),
    dataset_name: str = Form("2020_Historical_Baseline", min_length=1, max_length=120),
    fill_missing_with_synthetic: bool = Form(False),
    allow_duplicate: bool = Form(False),
    session: Session = Depends(get_db),
    user: User = Depends(ADMIN),
) -> dict[str, Any]:
    """Validate and register the historical (2020) reference population."""
    raw = await _read_csv(file)
    df, result = prepare_banking_frame(raw, fill_missing_with_synthetic)
    if not result.is_valid:
        raise _error(
            422,
            "Baseline validation failed: " + "; ".join(result.errors),
            "validation_error",
        )
    existing = store.find_baseline_by_hash(store.content_hash(df))
    if existing and not allow_duplicate:
        raise _error(
            409,
            f"An identical baseline is already registered as '{existing['name']}'.",
            "duplicate_dataset",
        )
    summary = store.register_baseline(
        session,
        user,
        df,
        dataset_name.strip(),
        PurePath(file.filename or "").name,
        result.synthetic_columns,
        bool(result.synthetic_columns),
    )
    store.audit(
        "baseline_registered",
        user,
        {
            "dataset_id": summary["dataset_id"],
            "name": summary["dataset_name"],
            "rows": summary["row_count"],
            "synthetic_columns": result.synthetic_columns,
        },
    )
    return {**summary, "warnings": result.warnings, "preview": result.preview}


@router.get("/datasets/baseline")
def get_baseline(
    session: Session = Depends(get_db), _: User = Depends(READ)
) -> dict[str, Any]:
    """Active baseline with model metrics and reference profile."""
    meta = store.current_baseline_meta()
    if not meta:
        return {
            "has_baseline": False,
            "message": "No historical baseline registered yet.",
        }
    summary = store.baseline_summary(session, meta)
    _, df = store.load_baseline()
    summary["preview"] = store.json_records(df, 10)
    return summary


# ---------------------------------------------------------------------------
# Production batches
# ---------------------------------------------------------------------------


@router.post("/batches/upload")
async def upload_batches(
    files: list[UploadFile] = File(...),
    batch_ids: list[str] | None = Form(None),
    batch_dates: list[str] | None = Form(None),
    descriptions: list[str] | None = Form(None),
    overwrite: bool = Form(False),
    allow_duplicate_content: bool = Form(False),
    fill_missing_with_synthetic: bool = Form(False),
    user: User = Depends(WRITE),
) -> dict[str, Any]:
    """Upload one or more production batch CSVs.

    Optional ``batch_ids``/``batch_dates``/``descriptions`` are parallel to
    ``files``. Each file is processed independently; per-file status is
    ``registered`` or ``rejected`` with reasons.
    """
    if len(files) > 24:
        raise _error(400, "At most 24 files can be uploaded at once.", "too_many_files")
    results = []
    seen: set[str] = set()
    for i, file in enumerate(files):
        filename = PurePath(file.filename or "").name
        batch_id = (
            batch_ids[i].strip()
            if batch_ids and i < len(batch_ids) and batch_ids[i].strip()
            else _batch_id_from_filename(filename)
        )
        batch_date = (
            batch_dates[i]
            if batch_dates and i < len(batch_dates) and batch_dates[i]
            else _infer_batch_date(filename)
        )
        description = (
            descriptions[i] if descriptions and i < len(descriptions) else ""
        )[:300]
        entry: dict[str, Any] = {"filename": filename, "batch_id": batch_id}
        try:
            if not BATCH_ID_PATTERN.match(batch_id):
                raise _error(422, f"Invalid batch_id '{batch_id}'.", "validation_error")
            if batch_id in seen:
                raise _error(
                    409,
                    f"batch_id '{batch_id}' appears twice in this upload.",
                    "duplicate_batch",
                )
            try:
                pd.Timestamp(batch_date or "invalid")
            except ValueError as exc:
                raise _error(
                    422,
                    f"batch_date '{batch_date}' is not a valid date (YYYY-MM-DD).",
                    "validation_error",
                ) from exc
            raw = await _read_csv(file)
            df, result = prepare_banking_frame(raw, fill_missing_with_synthetic)
            entry.update(
                {
                    "row_count": result.row_count,
                    "warnings": result.warnings,
                    "preview": result.preview[:5],
                    "labelled_rows": result.labelled_rows,
                }
            )
            if not result.is_valid:
                raise _error(422, "; ".join(result.errors), "validation_error")
            if store.get_batch(batch_id) and not overwrite:
                raise _error(
                    409,
                    f"Batch '{batch_id}' already exists; enable overwrite to replace it.",
                    "duplicate_batch",
                )
            twin = store.find_batch_by_hash(store.content_hash(df))
            if twin and twin["batch_id"] != batch_id and not allow_duplicate_content:
                raise _error(
                    409,
                    f"Identical content was already uploaded as batch '{twin['batch_id']}'.",
                    "duplicate_content",
                )
            meta = store.save_batch(
                user,
                batch_id,
                df,
                filename,
                str(pd.Timestamp(batch_date).date()),
                description,
                bool(result.synthetic_columns),
                result.synthetic_columns,
                overwrite=overwrite,
            )
            entry.update(
                {
                    "status": "registered",
                    "batch_date": meta["batch_date"],
                    "overwritten": bool(meta.get("overwritten")),
                }
            )
            store.audit(
                "batch_uploaded",
                user,
                {
                    "batch_id": batch_id,
                    "filename": filename,
                    "rows": meta["row_count"],
                    "overwrite": overwrite,
                },
            )
        except HTTPException as exc:
            entry.update(
                {
                    "status": "rejected",
                    "error": exc.detail["detail"],
                    "code": exc.detail["code"],
                }
            )
            store.audit(
                "batch_rejected",
                user,
                {
                    "batch_id": batch_id,
                    "filename": filename,
                    "reason": exc.detail["detail"],
                },
                level="warning",
            )
        seen.add(batch_id)
        results.append(entry)
    return {
        "registered_count": sum(r["status"] == "registered" for r in results),
        "rejected_count": sum(r["status"] == "rejected" for r in results),
        "results": results,
    }


@router.get("/batches")
def list_batches(
    risk_category: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    _: User = Depends(READ),
) -> list[dict[str, Any]]:
    """Batches with metadata, processing history and latest analysis summary."""
    rows = store.list_batches()
    if date_from:
        rows = [r for r in rows if r["batch_date"] >= date_from]
    if date_to:
        rows = [r for r in rows if r["batch_date"] <= date_to]
    if risk_category:
        rows = [
            r
            for r in rows
            if (r.get("latest_analysis") or {}).get("risk_category") == risk_category
        ]
    return rows


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, _: User = Depends(READ)) -> dict[str, Any]:
    """Batch metadata with a record preview."""
    meta = store.get_batch(batch_id)
    if not meta:
        raise _error(404, f"Batch '{batch_id}' not found.", "not_found")
    meta["uploaded_at"] = store.iso(meta.get("uploaded_at"))
    for h in meta.get("processing_history", []):
        h["at"] = (
            store.iso(h["at"]) if hasattr(h.get("at"), "isoformat") else h.get("at")
        )
    meta["preview"] = store.json_records(store.load_batch_records(batch_id), 10)
    return meta


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _run(
    session: Session, user: User, batch_ids: list[str], mode: str, force: bool
) -> dict[str, Any]:
    base_meta, ref_df = _baseline_or_409()
    metas = []
    for b in batch_ids:
        meta = store.get_batch(b)
        if not meta:
            raise _error(404, f"Batch '{b}' not found.", "not_found")
        metas.append(meta)
    signature = hashlib.sha256(
        "|".join(
            [
                mode,
                base_meta["content_sha256"],
                *sorted(m["content_sha256"] for m in metas),
            ]
        ).encode()
    ).hexdigest()
    if not force:
        prior = store.db().root_cause_reports.find_one(
            {"input_signature": signature}, {"_id": 0}, sort=[("analysed_at", -1)]
        )
        if prior:
            prior["reused"] = True
            if mode == "individual":
                store.db().production_batches.update_one(
                    {"batch_id": batch_ids[0]},
                    {"$set": {"latest_job_id": prior["job_id"], "status": "analysed"}},
                )
            return _clean_report(prior)

    frames = [store.load_batch_records(b) for b in batch_ids]
    prod_df = pd.concat(frames, ignore_index=True)
    if mode == "combined":
        job_batch_id = (
            "COMBINED-"
            + hashlib.sha1("|".join(sorted(batch_ids)).encode()).hexdigest()[:10]
        )
        name = "Combined: " + ", ".join(batch_ids)
    else:
        job_batch_id = batch_ids[0]
        name = batch_ids[0]
    dates = [m["batch_date"] for m in metas]
    model, metrics = store.load_champion(session, base_meta["model_id"])
    job = store.start_job(session, user, base_meta["model_id"], job_batch_id, dates)
    try:
        report = analyze_banking_drift(
            ref_df,
            prod_df,
            name,
            model,
            metrics,
            floor_key=f"ds{base_meta['dataset_id']}",
        )
        report.update(
            {
                "job_id": job.job_id,
                "analysis_type": mode,
                "batch_id": job_batch_id,
                "batch_ids": batch_ids,
                "batch_date": max(dates),
                "batch_dates": dates,
                "batches": [
                    {
                        k: m.get(k)
                        for k in (
                            "batch_id",
                            "batch_date",
                            "description",
                            "row_count",
                            "labelled_rows",
                            "filename",
                            "is_synthetic",
                            "scenario",
                        )
                    }
                    for m in metas
                ],
                "baseline_dataset_id": base_meta["dataset_id"],
                "baseline_name": base_meta["name"],
                "baseline_is_synthetic": base_meta.get("is_synthetic", False),
                "model_id": base_meta["model_id"],
                "is_synthetic": any(m.get("is_synthetic") for m in metas)
                or base_meta.get("is_synthetic", False),
                "input_signature": signature,
                "analysed_at": store.now(),
                "analysed_by": user.email,
                "reused": False,
            }
        )
        store.complete_job(session, job, base_meta["dataset_id"], report)
    except HTTPException:
        store.fail_job(session, job, "request error")
        raise
    except Exception as exc:
        logger.exception("Analysis failed for %s", batch_ids)
        store.fail_job(session, job, f"{type(exc).__name__}: {exc}")
        raise _error(
            500,
            f"Analysis failed: {type(exc).__name__}. Job {job.job_id} marked failed.",
            "analysis_failed",
        ) from exc
    store.record_batch_processing(batch_ids, job.job_id, user, mode)
    store.audit(
        "analysis_completed",
        user,
        {
            "job_id": job.job_id,
            "mode": mode,
            "batch_ids": batch_ids,
            "overall_dis": report["overall_dis"],
            "risk_category": report["risk_category"],
        },
    )
    return _clean_report(report)


@router.post("/analysis/run")
def run_analysis(
    body: AnalysisRequest,
    session: Session = Depends(get_db),
    user: User = Depends(WRITE),
) -> dict[str, Any]:
    """Individual mode: one report per batch. Combined mode: one pooled report.

    Already-processed identical inputs are returned (``reused=true``) unless
    ``force`` is set.
    """
    if body.mode == "combined":
        if len(body.batch_ids) < 2:
            raise _error(
                422, "Combined analysis needs at least two batches.", "validation_error"
            )
        return {
            "mode": "combined",
            "reports": [_run(session, user, body.batch_ids, "combined", body.force)],
        }
    return {
        "mode": "individual",
        "reports": [
            _run(session, user, [b], "individual", body.force) for b in body.batch_ids
        ],
    }


@router.post("/analysis/run-all")
def run_all(
    body: RunAllRequest, session: Session = Depends(get_db), user: User = Depends(WRITE)
) -> dict[str, Any]:
    """Individually analyse every registered batch."""
    ids = [b["batch_id"] for b in store.list_batches()]
    if not ids:
        raise _error(404, "No production batches are registered.", "not_found")
    reports = [_run(session, user, [b], "individual", body.force) for b in ids]
    return {
        "mode": "individual",
        "reports": [
            store._summary(r) | {"reused": r.get("reused", False)} for r in reports
        ],
    }


@router.get("/analysis/history")
def analysis_history(
    analysis_type: str | None = Query(None, pattern="^(individual|combined)$"),
    _: User = Depends(READ),
) -> dict[str, Any]:
    """All analyses plus the latest per batch and when drift first appeared."""
    runs = store.report_history(analysis_type)
    latest = [
        b["latest_analysis"] for b in store.list_batches() if b.get("latest_analysis")
    ]
    latest.sort(key=lambda r: (r["batch_date"] or "", r["batch_id"]))
    first_any = next((r for r in latest if r["drift_level"] != "low"), None)
    first_high = next((r for r in latest if r["drift_level"] == "high"), None)
    first_feature: dict[str, str] = {}
    for r in latest:
        full = store.get_report(r["job_id"]) or {}
        for f in full.get("feature_details", []):
            if f["severity"] == "high" and f["feature"] not in first_feature:
                first_feature[f["feature"]] = r["batch_id"]
    return {
        "runs": runs,
        "latest_per_batch": latest,
        "first_drift": {
            "first_moderate_or_higher": first_any
            and {
                "batch_id": first_any["batch_id"],
                "batch_date": first_any["batch_date"],
            },
            "first_high": first_high
            and {
                "batch_id": first_high["batch_id"],
                "batch_date": first_high["batch_date"],
            },
            "first_significant_by_feature": first_feature,
        },
    }


@router.get("/analysis/latest/{batch_id}")
def latest_for_batch(batch_id: str, _: User = Depends(READ)) -> dict[str, Any]:
    """Latest individual report for a batch."""
    meta = store.get_batch(batch_id)
    if not meta:
        raise _error(404, f"Batch '{batch_id}' not found.", "not_found")
    if not meta.get("latest_job_id"):
        raise _error(
            404, f"Batch '{batch_id}' has not been analysed yet.", "not_analysed"
        )
    return _clean_report(store.get_report(meta["latest_job_id"]))


@router.get("/analysis/{job_id}")
def get_analysis(job_id: int, _: User = Depends(READ)) -> dict[str, Any]:
    """Full stored report for a monitoring job."""
    report = store.get_report(job_id)
    if not report:
        raise _error(404, f"Analysis job {job_id} not found.", "not_found")
    return _clean_report(report)


@router.get("/analysis/{job_id}/features/{feature}")
def feature_detail(
    job_id: int, feature: str, _: User = Depends(READ)
) -> dict[str, Any]:
    """Detector values and overlaid distribution for one feature."""
    report = store.get_report(job_id)
    if not report:
        raise _error(404, f"Analysis job {job_id} not found.", "not_found")
    match = next(
        (f for f in report["feature_details"] if f["feature"] == feature), None
    )
    if not match:
        raise _error(404, f"Feature '{feature}' not in report.", "not_found")
    return {"feature": match, "distribution": report["distributions"].get(feature)}


@router.get("/recommendations/{job_id}")
def recommendations(job_id: int, _: User = Depends(READ)) -> dict[str, Any]:
    """Risk category, decision-engine output, actions and evidence for a job."""
    report = store.get_report(job_id)
    if not report:
        raise _error(404, f"Analysis job {job_id} not found.", "not_found")
    keys = (
        "job_id",
        "batch_id",
        "batch_name",
        "analysis_type",
        "risk_category",
        "drift_level",
        "performance_state",
        "alert_type",
        "alert_level",
        "recommendation",
        "recommended_actions",
        "evidence",
        "affected_features",
        "overall_dis",
        "is_synthetic",
    )
    out = {k: report.get(k) for k in keys}
    out["model_health_message"] = report["model_health"]["status_message"]
    return out


# ---------------------------------------------------------------------------
# Dashboard aggregates
# ---------------------------------------------------------------------------


def _arrays(df: pd.DataFrame, col: str) -> np.ndarray:
    v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return v[np.isfinite(v)]


@router.get("/overview")
def overview(
    session: Session = Depends(get_db), _: User = Depends(READ)
) -> dict[str, Any]:
    """KPIs, trends, heatmap and distributions for the banking overview."""
    meta = store.current_baseline_meta()
    batches = store.list_batches()
    latest = [b["latest_analysis"] for b in batches if b.get("latest_analysis")]
    latest.sort(key=lambda r: (r["batch_date"] or "", r["batch_id"]))
    newest = latest[-1] if latest else None
    reports = {r["job_id"]: store.get_report(r["job_id"]) for r in latest}

    labelled = [
        r
        for r in latest
        if reports[r["job_id"]]["model_health"].get("has_production_labels")
    ]
    if labelled:
        last = labelled[-1]
        health = {
            "status": last["performance_state"],
            "batch_id": last["batch_id"],
            "confirmed": True,
        }
    else:
        health = {"status": "unconfirmed", "batch_id": None, "confirmed": False}

    heat_features = MONITORED_FEATURES
    heatmap = {
        "features": [FEATURE_LABELS[f] for f in heat_features],
        "batches": [r["batch_id"] for r in latest],
        "d": [
            [
                next(
                    (
                        x["d"]
                        for x in reports[r["job_id"]]["feature_details"]
                        if x["feature"] == f
                    ),
                    None,
                )
                for r in latest
            ]
            for f in heat_features
        ],
        "dis": [
            [
                next(
                    (
                        x["dis"]
                        for x in reports[r["job_id"]]["feature_details"]
                        if x["feature"] == f
                    ),
                    None,
                )
                for r in latest
            ]
            for f in heat_features
        ],
    }

    distributions: dict[str, Any] = {}
    default_rates = []
    if meta:
        _, ref_df = store.load_baseline()
        frames = {"Baseline 2020": ref_df}
        for b in batches:
            frames[b["batch_id"]] = store.load_batch_records(b["batch_id"])
        for col in (
            "monthly_income",
            "credit_score",
            "outstanding_loan_amount",
            "credit_limit",
            "debt_to_income",
            "fd_balance",
        ):
            distributions[col] = numeric_distribution(
                {k: _arrays(v, col) for k, v in frames.items()}
            )
        distributions["employment_type"] = categorical_distribution(
            {
                k: v["employment_type"].dropna().astype(str).to_numpy(dtype=object)
                for k, v in frames.items()
            }
        )
        model_row = store.load_champion(session, meta["model_id"])[1] or {}
        lab = ref_df["loan_default"].dropna()
        default_rates.append(
            {
                "population": "Baseline 2020",
                "actual_default_rate": float(lab.mean()) if len(lab) else None,
                "predicted_default_rate": model_row.get("predicted_default_rate"),
                "labelled_rows": len(lab),
            }
        )
        for r in latest:
            dr = reports[r["job_id"]]["default_rates"]
            default_rates.append(
                {
                    "population": r["batch_id"],
                    "actual_default_rate": dr["production_actual"],
                    "predicted_default_rate": dr["production_predicted"],
                    "labelled_rows": reports[r["job_id"]]["model_health"].get(
                        "n_labels", 0
                    ),
                }
            )

    return {
        "baseline": (
            None
            if not meta
            else {
                "dataset_id": meta["dataset_id"],
                "name": meta["name"],
                "row_count": meta["row_count"],
                "is_synthetic": meta.get("is_synthetic", False),
                "created_at": store.iso(meta.get("created_at")),
            }
        ),
        "kpis": {
            "total_customers_monitored": int(sum(b["row_count"] for b in batches)),
            "production_batches": len(batches),
            "analysed_batches": len(latest),
            "features_monitored": len(MONITORED_FEATURES),
            "significant_drift_features": (
                newest["significant_drift_count"] if newest else None
            ),
            "drifting_features": newest["drifting_features_count"] if newest else None,
            "overall_dis": newest["overall_dis"] if newest else None,
            "latest_batch_id": newest["batch_id"] if newest else None,
            "latest_risk_category": newest["risk_category"] if newest else None,
            "model_health": health,
        },
        "trend": latest,
        "heatmap": heatmap,
        "distributions": distributions,
        "default_rates": default_rates,
        "any_synthetic": bool((meta or {}).get("is_synthetic"))
        or any(b.get("is_synthetic") for b in batches),
        "thresholds": {
            "batch_dis_moderate": settings.batch_dis_moderate,
            "batch_dis_high": settings.batch_dis_high,
            "severity_moderate": settings.severity_moderate,
            "severity_high": settings.severity_high,
        },
    }


@router.get("/model-health")
def model_health(
    session: Session = Depends(get_db), _: User = Depends(READ)
) -> dict[str, Any]:
    """Baseline model metrics and per-batch production evaluation."""
    meta = store.current_baseline_meta()
    if not meta:
        return {"has_baseline": False}
    summary = store.baseline_summary(session, meta)
    per_batch = []
    for b in store.list_batches():
        la = b.get("latest_analysis")
        if not la:
            continue
        rep = store.get_report(la["job_id"])
        per_batch.append(
            {
                "batch_id": b["batch_id"],
                "batch_date": b["batch_date"],
                "job_id": la["job_id"],
                "risk_category": la["risk_category"],
                "drift_level": la["drift_level"],
                "overall_dis": la["overall_dis"],
                **rep["model_health"],
            }
        )
    return {"has_baseline": True, "model": summary["model"], "batches": per_batch}


@router.get("/compare")
def compare_batches(
    batch_a: str = Query(...),
    batch_b: str = Query(...),
    _: User = Depends(READ),
) -> dict[str, Any]:
    """Compare two batches: their baseline-relative results and direct A→B drift."""
    if batch_a == batch_b:
        raise _error(422, "Choose two different batches.", "validation_error")
    out: dict[str, Any] = {"batch_a": batch_a, "batch_b": batch_b}
    frames = {}
    for key, b in (("a", batch_a), ("b", batch_b)):
        meta = store.get_batch(b)
        if not meta:
            raise _error(404, f"Batch '{b}' not found.", "not_found")
        frames[key] = store.load_batch_records(b)
        rep = (
            store.get_report(meta["latest_job_id"])
            if meta.get("latest_job_id")
            else None
        )
        out[f"summary_{key}"] = store._summary(rep)
        out[f"features_{key}"] = {
            f["feature"]: {
                k: f.get(k)
                for k in (
                    "psi",
                    "ks_stat",
                    "ks_p",
                    "jsd",
                    "d",
                    "dis",
                    "severity",
                    "drift_status",
                )
            }
            for f in (rep or {}).get("feature_details", [])
        }
    direct = []
    dists = {}
    for feat in MONITORED_FEATURES:
        kind = "numerical" if feat in NUMERICAL_FEATURES else "categorical"
        a, b = frames["a"][feat], frames["b"][feat]
        if kind == "numerical":
            av, bv = _arrays(frames["a"], feat), _arrays(frames["b"], feat)
            dists[feat] = numeric_distribution({batch_a: av, batch_b: bv})
        else:
            av = a.dropna().astype(str).to_numpy(dtype=object)
            bv = b.dropna().astype(str).to_numpy(dtype=object)
            dists[feat] = categorical_distribution({batch_a: av, batch_b: bv})
        floors = (
            noise_floors(f"batch:{batch_a}:{feat}", av, len(bv), kind)
            if len(av) >= 30 and len(bv) >= 30
            else None
        )
        res = compute_feature_drift(a, b, feat, kind, floors)
        direct.append(
            {
                k: res.get(k)
                for k in (
                    "feature",
                    "label",
                    "type",
                    "psi",
                    "ks_stat",
                    "ks_p",
                    "jsd",
                    "d",
                    "severity",
                    "drift_status",
                    "reference_stats",
                    "production_stats",
                )
            }
        )
    out["direct_drift"] = direct
    out["distributions"] = dists
    return out


# ---------------------------------------------------------------------------
# Reports and audit
# ---------------------------------------------------------------------------


@router.get("/reports/{job_id}/pdf")
def report_pdf(job_id: int, user: User = Depends(READ)) -> Response:
    """Risk-committee PDF for a completed analysis."""
    report = store.get_report(job_id)
    if not report:
        raise _error(404, f"Analysis job {job_id} not found.", "not_found")
    pdf = generate_committee_pdf(_clean_report(report))
    store.audit(
        "report_generated", user, {"job_id": job_id, "batch_id": report["batch_id"]}
    )
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in report["batch_id"])
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="DriftGuard_Risk_Report_{safe}_job{job_id}.pdf"'
        },
    )


@router.get("/audit-logs")
def audit_logs(
    limit: int = Query(200, ge=1, le=1000),
    event: str | None = Query(None, max_length=60),
    session: Session = Depends(get_db),
    _: User = Depends(READ),
) -> dict[str, Any]:
    """Audit events (MongoDB) and monitoring jobs with alerts (PostgreSQL)."""
    return {
        "events": store.audit_events(limit, event),
        "jobs": store.job_rows(session, limit),
    }


# ---------------------------------------------------------------------------
# Synthetic data (clearly labelled)
# ---------------------------------------------------------------------------


@router.get("/simulation/scenarios")
def scenarios(_: User = Depends(READ)) -> dict[str, Any]:
    """Catalogue of synthetic production scenarios."""
    return SCENARIOS


def _generate_batch(user: User, body: GenerateBatchRequest) -> dict[str, Any]:
    if store.get_batch(body.batch_id) and not body.overwrite:
        raise _error(
            409,
            f"Batch '{body.batch_id}' already exists; enable overwrite to replace it.",
            "duplicate_batch",
        )
    seed = (
        body.seed
        if body.seed is not None
        else int(hashlib.sha1(body.batch_id.encode()).hexdigest()[:6], 16)
    )
    gen = BankingDataGenerator(seed=seed)
    raw = gen.generate_production_batch(
        body.batch_id, body.n, body.scenario, body.with_labels, body.concept_shift
    )
    df, result = prepare_banking_frame(raw)
    desc = (
        body.description
        or f"SYNTHETIC scenario {body.scenario}: {SCENARIOS[body.scenario]['name']}"
        + (" + concept shift" if body.concept_shift else "")
    )
    meta = store.save_batch(
        user,
        body.batch_id,
        df,
        None,
        body.batch_date.isoformat(),
        desc,
        True,
        list(df.columns),
        scenario=body.scenario + ("+concept" if body.concept_shift else ""),
        overwrite=body.overwrite,
    )
    store.audit(
        "synthetic_batch_generated",
        user,
        {
            "batch_id": body.batch_id,
            "scenario": body.scenario,
            "rows": body.n,
            "labels": body.with_labels,
            "concept_shift": body.concept_shift,
        },
    )
    return {
        "status": "registered",
        "batch_id": body.batch_id,
        "row_count": meta["row_count"],
        "is_synthetic": True,
        "scenario": meta["scenario"],
        "warnings": result.warnings,
    }


@router.post("/simulation/generate-batch", status_code=201)
def generate_batch(
    body: GenerateBatchRequest, user: User = Depends(WRITE)
) -> dict[str, Any]:
    """Generate and register a SYNTHETIC production batch for a scenario."""
    return _generate_batch(user, body)


def _generate_baseline(
    session: Session, user: User, body: GenerateBaselineRequest
) -> dict[str, Any]:
    gen = BankingDataGenerator(seed=body.seed)
    df, _ = prepare_banking_frame(gen.generate_baseline_2020(body.n))
    existing = store.find_baseline_by_hash(store.content_hash(df))
    if existing:
        return {**store.baseline_summary(session, existing), "reused": True}
    summary = store.register_baseline(
        session, user, df, body.dataset_name, None, list(df.columns), True
    )
    store.audit(
        "synthetic_baseline_generated",
        user,
        {"dataset_id": summary["dataset_id"], "rows": body.n},
    )
    return {**summary, "reused": False}


@router.post("/simulation/generate-baseline", status_code=201)
def generate_baseline(
    body: GenerateBaselineRequest,
    session: Session = Depends(get_db),
    user: User = Depends(ADMIN),
) -> dict[str, Any]:
    """Generate and register a SYNTHETIC 2020 baseline (reused if identical)."""
    return _generate_baseline(session, user, body)


DEMO_BATCHES = [
    ("January_2026", "2026-01-31", "A", 1000, False, False),
    ("March_2026", "2026-03-31", "B", 1200, False, False),
    ("June_2026", "2026-06-30", "C", 1500, False, False),
    ("September_2026", "2026-09-30", "D", 1800, True, False),
]
CONCEPT_BATCH = ("November_2026", "2026-11-15", "D", 1600, True, True)


@router.post("/simulation/seed-demo")
def seed_demo(
    body: SeedDemoRequest,
    session: Session = Depends(get_db),
    user: User = Depends(ADMIN),
) -> dict[str, Any]:
    """Seed the SYNTHETIC demo: 2020 baseline, 2026 batches, individual + combined analyses."""
    baseline = _generate_baseline(session, user, GenerateBaselineRequest())
    plan = DEMO_BATCHES + ([CONCEPT_BATCH] if body.include_concept_shift_batch else [])
    created = []
    for i, (bid, bdate, scenario, n, labels, concept) in enumerate(plan):
        if store.get_batch(bid) and not body.force:
            created.append({"batch_id": bid, "status": "exists"})
            continue
        _generate_batch(
            user,
            GenerateBatchRequest(
                scenario=scenario,
                batch_id=bid,
                batch_date=bdate,
                n=n,
                with_labels=labels,
                concept_shift=concept,
                seed=1000 + i,
                overwrite=True,
            ),
        )
        created.append({"batch_id": bid, "status": "generated"})
    ids = [p[0] for p in plan]
    reports = [_run(session, user, [b], "individual", body.force) for b in ids]
    combined = _run(session, user, ids[:4], "combined", body.force)
    store.audit("demo_seeded", user, {"batches": ids})
    return {
        "baseline": {
            "dataset_id": baseline["dataset_id"],
            "name": baseline["dataset_name"],
            "reused": baseline.get("reused"),
        },
        "batches": created,
        "analyses": [store._summary(r) for r in reports],
        "combined": store._summary(combined),
        "synthetic": True,
    }
