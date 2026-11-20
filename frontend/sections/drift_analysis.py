"""C. Customer Drift Analysis: PSI, KS, JSD and distributions per financial attribute."""

from typing import Any

import api_client as api
import pandas as pd
import streamlit as st
from components import (
    chart,
    distribution_figure,
    fmt,
    header,
    safe,
    status_cell,
    table,
)

from sections.shared import _summary_row, can_write, feature_table

KEY_FEATURES = [
    ("monthly_income", "Monthly Income"),
    ("credit_score", "Credit Score"),
    ("credit_limit", "Credit Limit"),
    ("fd_balance", "FD Balance"),
    ("outstanding_loan_amount", "Outstanding Loans"),
    ("debt_to_income", "Debt-to-Income Ratio"),
]


def render() -> None:
    header(
        "Customer Distribution Drift Analysis",
        "Statistical tests (PSI, Kolmogorov-Smirnov, Jensen-Shannon Divergence) across applicant attributes",
    )
    batches = safe(lambda: api.get("/batches"), []) or []
    if not batches:
        st.info("Upload production batches in Dataset Management first.")
        return
    ids = [b["batch_id"] for b in batches]
    by_id = {b["batch_id"]: b for b in batches}
    default = st.session_state.get("drift_batch", ids[-1])

    def format_batch(batch_id: str) -> str:
        b = by_id.get(batch_id, {})
        drift_lvl = b.get("drift_level")
        dot = (
            "🔴"
            if drift_lvl == "high"
            else ("🟡" if drift_lvl == "moderate" else ("🟢" if drift_lvl == "low" else "⚪"))
        )
        count = b.get("row_count", 0)
        status_txt = f"{drift_lvl.upper()} DRIFT" if drift_lvl else "PENDING ANALYSIS"
        return f"{dot} {batch_id}  ·  {count:,} customers  ·  {status_txt}"

    bid = st.selectbox(
        "Select Production Batch to Inspect",
        ids,
        index=ids.index(default) if default in ids else len(ids) - 1,
        format_func=format_batch,
        key="drift_batch_sel",
    )
    st.session_state["drift_batch"] = bid

    if by_id[bid].get("latest_job_id") is None:
        st.info(f"Batch **{bid}** has not been analysed yet.")
        if st.button("Run Drift Analysis", type="primary", disabled=not can_write()):
            with st.spinner("Computing PSI, KS, JSD and DIS in backend…"):
                safe(lambda: api.post("/analysis/run", json={"batch_ids": [bid]}))
            st.rerun()
        return

    report = safe(lambda: api.get(f"/analysis/latest/{bid}"))
    if not report:
        return

    _summary_row(report)
    _drift_types(report)

    tab_key, tab_table, tab_inspect = st.tabs(
        [
            "📊 Core Banking Attributes",
            "📋 Comprehensive Feature Metrics",
            "🔬 Attribute Deep-Dive Inspector",
        ]
    )

    with tab_key:
        _key_feature_grid(report)

    with tab_table:
        _feature_results(report)

    with tab_inspect:
        _feature_inspector(report)


def _drift_types(report: dict[str, Any]) -> None:
    """Data drift / prediction drift / concept drift / degradation, kept separate."""
    t = report.get("drift_types")
    if not t:
        return

    def show(v: Any) -> str:
        return "Unknown (no labels)" if v is None else ("Yes" if v else "No")

    cols = st.columns(4)
    cols[0].metric("Data drift", show(t["data_drift"]))
    cols[1].metric("Prediction drift (ADWIN / KS)", show(t["prediction_drift"]))
    cols[2].metric("Concept drift suspected", show(t["concept_drift_suspected"]))
    cols[3].metric("Confirmed degradation", show(t["confirmed_degradation"]))
    streams = (report.get("model_health") or {}).get("streams") or {}
    pred = streams.get("prediction") or {}
    if pred.get("detected"):
        st.caption(
            f"ADWIN detected a change in the predicted-probability stream at production record "
            f"{pred['first_detection_index']} (reference mean {pred['reference_mean']:.3f} → "
            f"production mean {pred['production_mean']:.3f})."
        )
    for n in t.get("notes", []):
        st.caption(n)
    if report.get("leakage_warning"):
        st.warning(
            "The current champion was trained on part of: "
            + ", ".join(report["leakage_warning"])
            + ". Performance on these batches is optimistic."
        )


def _key_feature_grid(report: dict[str, Any]) -> None:
    st.markdown("##### Core Financial Attributes: 2020 Baseline vs Production Batch")
    by_feature = {f["feature"]: f for f in report["feature_details"]}
    names = {"baseline": "Baseline 2020", "production": report["batch_name"]}
    for row in range(2):
        cols = st.columns(3)
        for col, (feature, label) in zip(
            cols, KEY_FEATURES[row * 3 : row * 3 + 3], strict=True
        ):
            f = by_feature[feature]
            psi = fmt(f["psi"], ".3f")
            with col:
                chart(
                    distribution_figure(
                        report["distributions"].get(feature),
                        f"{label} · {f['drift_status']} (PSI {psi})",
                        270,
                        names=names,
                        legend=row == 0 and col is cols[0],
                    ),
                    key=f"key_{feature}",
                )


def _feature_results(report: dict[str, Any]) -> None:
    st.markdown("##### Feature-Level Drift Diagnostics")
    df = feature_table(report)
    table(
        df,
        {
            "PSI": ".4f",
            "KS Statistic": ".4f",
            "KS p-value": ".2e",
            "JSD": ".4f",
            "D (normalised)": ".2f",
            "DIS": ".3f",
        },
        {"Drift Status": status_cell},
        height=460,
    )
    t = report["thresholds"]
    st.caption(
        f"<b>Thresholds:</b> Stable (D < {t['severity_moderate']}) · Moderate ({t['severity_moderate']} ≤ D < {t['severity_high']}) · Significant (D ≥ {t['severity_high']}). "
        "Normalised drift D subtracts bootstrap noise floor at 95th percentile before scaling. "
        "Conventional alerts flag PSI > 0.25, KS p < 0.01, or JSD > 0.1."
    )


def _feature_inspector(report: dict[str, Any]) -> None:
    st.markdown("##### Detailed Feature Distribution & Statistics")
    feats = {f["label"]: f for f in report["feature_details"]}
    ranked = sorted(feats, key=lambda k: feats[k].get("dis", 0), reverse=True)
    label = st.selectbox(
        "Select Feature to Examine", ranked, key=f"feat_{report['job_id']}"
    )
    f = feats[label]
    left, right = st.columns([3, 2])
    with left:
        chart(
            distribution_figure(
                report["distributions"].get(f["feature"]),
                f"{label}: 2020 Baseline vs Production Distribution",
                370,
                names={"baseline": "Baseline 2020", "production": "Production"},
            ),
            key="feat_dist",
        )
    with right:
        st.markdown(
            f"### {label} &nbsp; {_status_html(f['drift_status'])}",
            unsafe_allow_html=True,
        )
        det = pd.DataFrame(
            [
                {
                    "Detector": "PSI",
                    "Value": f["psi"],
                    "Noise floor": (f["floors"] or {}).get("psi"),
                    "Normalised D": (f["normalised"] or {}).get("psi"),
                },
                {
                    "Detector": "KS statistic",
                    "Value": f["ks_stat"],
                    "Noise floor": (f["floors"] or {}).get("ks"),
                    "Normalised D": (f["normalised"] or {}).get("ks"),
                },
                {
                    "Detector": "JSD",
                    "Value": f["jsd"],
                    "Noise floor": (f["floors"] or {}).get("jsd"),
                    "Normalised D": (f["normalised"] or {}).get("jsd"),
                },
            ]
        )
        table(
            det,
            {"Value": ".4f", "Noise floor": ".4f", "Normalised D": ".2f"},
            missing="n/a",
        )
        if f["ks_p"] is not None:
            st.caption(f"Kolmogorov-Smirnov p-value: `{f['ks_p']:.2e}`")
        if f["type"] == "numerical" and f["reference_stats"]:
            rs, ps = f["reference_stats"], f["production_stats"]
            stats = pd.DataFrame({"Baseline": rs, "Production": ps}).loc[
                ["mean", "median", "std", "p05", "p95", "missing_rate"]
            ]
            stats.index = [
                "Mean",
                "Median",
                "Std Dev",
                "5th Percentile",
                "95th Percentile",
                "Missing Rate",
            ]
            stats["Relative Change"] = [
                (
                    ((p - b) / b if (b and p is not None) else None)
                    if k != "Missing Rate"
                    else None
                )
                for k, b, p in zip(
                    stats.index, stats["Baseline"], stats["Production"], strict=True
                )
            ]
            table(
                stats.reset_index(names="Metric"),
                {"Baseline": ",.2f", "Production": ",.2f", "Relative Change": "+.1%"},
            )
        elif f["type"] == "categorical":
            if f.get("new_categories"):
                st.warning(
                    "⚠️ New categories observed in production: "
                    + ", ".join(f["new_categories"])
                )
            if f.get("missing_categories"):
                st.info(
                    "ℹ️ Baseline categories absent in production: "
                    + ", ".join(f["missing_categories"])
                )


def _status_html(status: str) -> str:
    from components import STATUS, badge

    fg, bg = STATUS.get(status, ("#0F172A", "#FFFFFF"))
    return badge(status, fg, bg, dot=True)
