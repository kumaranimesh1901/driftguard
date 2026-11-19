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
    ("monthly_income", "Monthly income"),
    ("credit_score", "Credit score"),
    ("credit_limit", "Credit limit"),
    ("fd_balance", "FD balance"),
    ("outstanding_loan_amount", "Outstanding loans"),
    ("debt_to_income", "Debt-to-income ratio"),
]


def render() -> None:
    header(
        "Customer Drift Analysis",
        "How each financial attribute of a 2026 batch differs from the 2020 baseline (PSI, KS, JSD)",
    )
    batches = safe(lambda: api.get("/batches"), []) or []
    if not batches:
        st.info("Upload production batches in Dataset Management first.")
        return
    ids = [b["batch_id"] for b in batches]
    by_id = {b["batch_id"]: b for b in batches}
    default = st.session_state.get("drift_batch", ids[-1])
    bid = st.selectbox(
        "Production batch",
        ids,
        index=ids.index(default) if default in ids else len(ids) - 1,
        key="drift_batch_sel",
    )
    st.session_state["drift_batch"] = bid
    if by_id[bid].get("latest_job_id") is None:
        st.info(f"{bid} has not been analysed yet.")
        if st.button("Run analysis", type="primary", disabled=not can_write()):
            with st.spinner("Computing drift, DIS and model health in the backend…"):
                safe(lambda: api.post("/analysis/run", json={"batch_ids": [bid]}))
            st.rerun()
        return
    report = safe(lambda: api.get(f"/analysis/latest/{bid}"))
    if not report:
        return
    _summary_row(report)
    _key_feature_grid(report)
    _feature_results(report)


def _key_feature_grid(report: dict[str, Any]) -> None:
    st.markdown("#### Key financial attributes: 2020 baseline vs this batch")
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
                        260,
                        names=names,
                        legend=row == 0 and col is cols[0],
                    ),
                    key=f"key_{feature}",
                )


def _feature_results(report: dict[str, Any]) -> None:
    st.markdown("#### Feature-level results")
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
        f"Status from normalised drift D: Stable < {t['severity_moderate']} ≤ Moderate < {t['severity_high']} ≤ Significant. "
        "D removes each detector's bootstrap no-drift noise floor (95th percentile) before scaling. "
        "'Conventional alert' = PSI > 0.25, KS p < 0.01 or JSD > 0.1, shown for comparison — with large batches KS "
        "flags even negligible shifts."
    )

    st.markdown("#### Inspect a feature")
    feats = {f["label"]: f for f in report["feature_details"]}
    ranked = sorted(feats, key=lambda k: feats[k].get("dis", 0), reverse=True)
    label = st.selectbox("Feature", ranked, key=f"feat_{report['job_id']}")
    f = feats[label]
    left, right = st.columns([3, 2])
    with left:
        chart(
            distribution_figure(
                report["distributions"].get(f["feature"]),
                f"{label}: baseline vs production",
                360,
                names={"baseline": "Baseline 2020", "production": "Production"},
            ),
            key="feat_dist",
        )
    with right:
        st.markdown(
            f"**{label}** &nbsp; {_status_html(f['drift_status'])}",
            unsafe_allow_html=True,
        )
        det = pd.DataFrame(
            [
                {
                    "Detector": "PSI",
                    "Value": f["psi"],
                    "Noise floor": (f["floors"] or {}).get("psi"),
                    "Normalised": (f["normalised"] or {}).get("psi"),
                },
                {
                    "Detector": "KS statistic",
                    "Value": f["ks_stat"],
                    "Noise floor": (f["floors"] or {}).get("ks"),
                    "Normalised": (f["normalised"] or {}).get("ks"),
                },
                {
                    "Detector": "JSD",
                    "Value": f["jsd"],
                    "Noise floor": (f["floors"] or {}).get("jsd"),
                    "Normalised": (f["normalised"] or {}).get("jsd"),
                },
            ]
        )
        table(
            det,
            {"Value": ".4f", "Noise floor": ".4f", "Normalised": ".2f"},
            missing="n/a",
        )
        if f["ks_p"] is not None:
            st.caption(f"KS p-value {f['ks_p']:.2e}")
        if f["type"] == "numerical" and f["reference_stats"]:
            rs, ps = f["reference_stats"], f["production_stats"]
            stats = pd.DataFrame({"Baseline": rs, "Production": ps}).loc[
                ["mean", "median", "std", "p05", "p95", "missing_rate"]
            ]
            stats.index = [
                "Mean",
                "Median",
                "Std dev",
                "5th pct",
                "95th pct",
                "Missing rate",
            ]
            stats["Change"] = [
                (
                    ((p - b) / b if (b and p is not None) else None)
                    if k != "Missing rate"
                    else None
                )
                for k, b, p in zip(
                    stats.index, stats["Baseline"], stats["Production"], strict=True
                )
            ]
            table(
                stats.reset_index(names="Statistic"),
                {"Baseline": ",.3f", "Production": ",.3f", "Change": "+.1%"},
            )
        elif f["type"] == "categorical":
            if f.get("new_categories"):
                st.warning(
                    "New categories in production: " + ", ".join(f["new_categories"])
                )
            if f.get("missing_categories"):
                st.info(
                    "Baseline categories absent from production: "
                    + ", ".join(f["missing_categories"])
                )


def _status_html(status: str) -> str:
    from components import STATUS, badge

    fg, bg = STATUS.get(status, ("#1B2433", "#fff"))
    return badge(status, fg, bg)
