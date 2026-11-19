"""Helpers shared by several dashboard sections."""

from typing import Any

import api_client as api
import pandas as pd
import streamlit as st
from components import RISK, fmt, kpi, safe, synthetic_notice

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


def analysis_choices(include_combined: bool = True) -> list[dict[str, Any]]:
    """Latest individual analysis per batch, then combined runs (newest first)."""
    hist = safe(lambda: api.get("/analysis/history"), {}) or {}
    latest = hist.get("latest_per_batch", [])
    out = [
        dict(r, _label=f"{r['batch_id']} · {r['batch_date']} · job #{r['job_id']}")
        for r in latest
    ]
    if include_combined:
        combined = [r for r in hist.get("runs", []) if r["analysis_type"] == "combined"]
        combined.sort(key=lambda r: r["job_id"], reverse=True)
        out += [
            dict(
                r,
                _label=f"Combined ({len(r['batch_ids'])} batches: {', '.join(r['batch_ids'])}) · job #{r['job_id']}",
            )
            for r in combined
        ]
    return out


def select_analysis(
    key: str, include_combined: bool = True, label: str = "Analysis"
) -> dict[str, Any] | None:
    """Selectbox over analyses; remembers the last choice across pages."""
    choices = analysis_choices(include_combined)
    if not choices:
        st.info(
            "No analyses yet. Upload production batches and run an analysis in **Dataset Management → Select & analyse**, "
            "or seed the synthetic demo from the sidebar."
        )
        return None
    labels = [c["_label"] for c in choices]
    remembered = st.session_state.get("selected_job")
    index = next((i for i, c in enumerate(choices) if c["job_id"] == remembered), None)
    if index is None:  # default: newest individual batch
        individual = [
            i for i, c in enumerate(choices) if c["analysis_type"] == "individual"
        ]
        index = individual[-1] if individual else 0
    picked = st.selectbox(label, labels, index=index, key=key)
    choice = choices[labels.index(picked)]
    st.session_state["selected_job"] = choice["job_id"]
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
                "Conventional alert": "Yes" if f["conventional_alert"] else "No",
            }
        )
    return pd.DataFrame(rows)


def _summary_row(report: dict[str, Any]) -> None:
    synthetic_notice(report.get("is_synthetic"))
    c = st.columns(5)
    kpi(
        c[0],
        "Customers compared",
        f"{report['customer_count']:,}",
        f"vs {report['baseline_count']:,} baseline",
    )
    kpi(
        c[1],
        "Drifting features",
        f"{report['drifting_features_count']} / {report['features_monitored']}",
        f"{report['significant_drift_count']} significant",
    )
    kpi(
        c[2],
        "Overall DIS",
        fmt(report["overall_dis"]),
        f"drift level: {report['drift_level']}",
        accent=RISK[report["risk_category"]][0],
    )
    kpi(
        c[3],
        "Risk category",
        report["risk_category"].replace(" RISK", ""),
        report["alert_type"] or "no alert",
        accent=RISK[report["risk_category"]][0],
    )
    kpi(
        c[4],
        "Performance",
        report["performance_state"].title(),
        (
            "labels confirmed"
            if report["model_health"].get("has_production_labels")
            else "labels pending"
        ),
    )
    st.caption(
        f"Job #{report['job_id']} · {report['analysis_type']} analysis · analysed {report['analysed_at']} by "
        f"{report['analysed_by']}"
        + (" · reused stored result" if report.get("reused") else "")
    )
