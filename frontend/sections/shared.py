"""Helpers shared by several dashboard sections."""

from typing import Any

import api_client as api
import pandas as pd
import streamlit as st
from components import (
    RISK,
    fmt,
    format_perf_state,
    format_timestamp,
    kpi,
    safe,
    synthetic_notice,
)

FEATURE_LABELS = {
    "monthly_income": "Monthly Income",
    "credit_score": "Credit Score",
    "credit_limit": "Credit Limit",
    "outstanding_loan_amount": "Outstanding Loan Amount",
    "fd_balance": "FD Balance",
    "debt_to_income": "Debt-to-Income",
    "loan_amount_requested": "Loan Amount Requested",
    "interest_rate": "Interest Rate",
    "number_of_existing_loans": "Existing Loans",
    "loan_tenure": "Loan Tenure (months)",
    "age": "Age",
    "employment_type": "Employment Type",
}


def role() -> str:
    """Signed-in role."""
    return st.session_state.get("role", "reviewer")


def can_write() -> bool:
    """Analysts and administrators may upload and analyse."""
    return role() in {"analyst", "administrator"}


def is_admin() -> bool:
    """Administrators may register baselines and seed the demo."""
    return role() == "administrator"


def is_test_batch(batch_id: str | None) -> bool:
    """Identify UI/automated test batches by ID prefix (never deletes data)."""
    return bool(batch_id and str(batch_id).startswith("UI_Test_"))


def hide_test_batches() -> bool:
    """Whether UI_Test_* batches are currently hidden in UI views."""
    return bool(st.session_state.get("hide_test_batches", True))


def visible_batches(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter batch list according to the 'Hide test batches' toggle."""
    if not hide_test_batches():
        return list(batches)
    return [b for b in batches if not is_test_batch(b.get("batch_id"))]


def latest_batch(batches: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the latest batch by batch_date, breaking ties by registration time."""
    if not batches:
        return None
    return max(
        batches,
        key=lambda b: (
            b.get("batch_date") or "",
            b.get("uploaded_at") or b.get("created_at") or "",
            b.get("batch_id") or "",
        ),
    )


def analysis_choices(include_combined: bool = True) -> list[dict[str, Any]]:
    """Latest individual analysis per visible batch, then combined runs (newest first)."""
    hist = safe(lambda: api.get("/analysis/history"), {}) or {}
    latest = [
        r
        for r in hist.get("latest_per_batch", [])
        if not (hide_test_batches() and is_test_batch(r.get("batch_id")))
    ]

    def _format_label(r: dict[str, Any]) -> str:
        risk = r.get("risk_category") or "LOW RISK"
        dis = r.get("overall_dis")
        dis_str = f"DIS {dis:.3f}" if dis is not None else "DIS —"
        return f"{r['batch_id']} · {risk} · {dis_str} · Job #{r['job_id']}"

    out = [dict(r, _label=_format_label(r)) for r in latest]
    if include_combined:
        combined = [r for r in hist.get("runs", []) if r["analysis_type"] == "combined"]
        combined.sort(key=lambda r: r["job_id"], reverse=True)
        out += [
            dict(
                r,
                _label=f"Combined ({len(r['batch_ids'])} batches: {', '.join(r['batch_ids'])}) · Job #{r['job_id']}",
            )
            for r in combined
        ]
    return out


def select_analysis(
    key: str,
    include_combined: bool = True,
    label: str = "Analysis run",
) -> dict[str, Any] | None:
    """Resolve the active analysis synced with the global batch selector (or combined)."""
    choices = analysis_choices(include_combined)
    if not choices:
        st.info(
            "No analyses yet. Upload production batches and run an analysis in **Dataset Management → Select & analyse**, "
            "or seed the synthetic demo from Dataset Management."
        )
        return None

    # Prefer the globally selected batch from page_header when available.
    selected_bid = st.session_state.get("selected_batch_id")
    remembered = st.session_state.get("selected_job")
    index = next(
        (
            i
            for i, c in enumerate(choices)
            if c.get("analysis_type") == "individual" and c.get("batch_id") == selected_bid
        ),
        None,
    )
    if index is None and remembered is not None:
        index = next((i for i, c in enumerate(choices) if c["job_id"] == remembered), None)
    if index is None:
        individual = [
            i for i, c in enumerate(choices) if c["analysis_type"] == "individual"
        ]
        index = individual[-1] if individual else 0

    if not include_combined:
        choice = choices[index]
        st.session_state["selected_job"] = choice["job_id"]
        return choice

    labels = [c["_label"] for c in choices]
    picked = st.selectbox(label, labels, index=index, key=key)
    choice = choices[labels.index(picked)]
    st.session_state["selected_job"] = choice["job_id"]
    if choice.get("analysis_type") == "individual":
        st.session_state["selected_batch_id"] = choice["batch_id"]
    return choice


def feature_table(report: dict[str, Any]) -> pd.DataFrame:
    """Feature-level results as a display DataFrame."""
    rows = []
    for f in report["feature_details"]:
        rows.append(
            {
                "Feature": f["label"],
                "PSI": f["psi"],
                "KS Statistic": f["ks_stat"],
                "KS p-value": f["ks_p"],
                "JSD": f["jsd"],
                "D (normalised)": f["d"],
                "DIS": f.get("dis"),
                "Drift Status": f["drift_status"],
                "Alert Triggered": "Yes" if f["conventional_alert"] else "No",
            }
        )
    return pd.DataFrame(rows)


def _summary_row(report: dict[str, Any]) -> None:
    synthetic_notice(report.get("is_synthetic"))
    risk_cat = report.get("risk_category", "LOW RISK")
    accent_color = RISK.get(risk_cat, ("#2563EB", "#EFF6FF"))[0]
    perf_state = report.get("performance_state") or "unknown"
    has_labels = bool((report.get("model_health") or {}).get("has_production_labels"))

    c = st.columns(5)
    kpi(
        c[0],
        "Customers Compared",
        f"{report['customer_count']:,}",
        f"vs {report['baseline_count']:,} baseline",
    )
    kpi(
        c[1],
        "Drifting Attributes",
        f"{report['drifting_features_count']} / {report['features_monitored']}",
        f"{report['significant_drift_count']} significant",
        accent=accent_color if report["drifting_features_count"] else "#16A34A",
    )
    kpi(
        c[2],
        "Overall DIS",
        fmt(report["overall_dis"]),
        f"drift: {report['drift_level'].upper()}",
        accent=accent_color,
    )
    kpi(
        c[3],
        "Risk Category",
        risk_cat.replace(" RISK", ""),
        report["alert_type"] or "standard monitoring",
        accent=accent_color,
    )
    kpi(
        c[4],
        "Model Performance",
        "Unknown" if not has_labels else perf_state.title(),
        "Unknown – labels pending" if not has_labels else "labels confirmed",
        accent="#64748B" if not has_labels else ("#16A34A" if perf_state == "stable" else accent_color),
    )
    st.caption(
        f"Analysis Job #{report['job_id']} · {report['analysis_type'].title()} · "
        f"{format_timestamp(report.get('analysed_at'))} by {report['analysed_by']}"
        + (" (cached)" if report.get("reused") else "")
    )
    _ = format_perf_state  # exported for sections
