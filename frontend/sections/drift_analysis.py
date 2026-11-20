"""C. Customer Drift Analysis: summary, drift types, progress-bar feature table, detail inspector."""

from typing import Any

import api_client as api
import pandas as pd
import streamlit as st
import theme
from components import (
    chart,
    distribution_figure,
    empty_state,
    fmt,
    info_callout,
    page_header,
    safe,
    section,
    status_pill,
    table,
)

from sections.shared import _summary_row, can_write


def render() -> None:
    selected_meta = page_header(
        "Customer Distribution Drift Analysis",
        "Statistical tests (PSI, KS, JSD) vs bootstrap noise floors and ADWIN stream monitoring",
        show_batch_selector=True,
    )
    if not selected_meta:
        empty_state(
            "NO BATCHES",
            "No production batches available",
            "Upload or generate production batches in Dataset Management first.",
        )
        return
    bid = selected_meta["batch_id"]

    if selected_meta.get("latest_job_id") is None:
        empty_state(
            "PENDING ANALYSIS",
            f"Batch {bid} has not been analysed yet",
            "Run drift analysis to compute PSI, KS, JSD, DIS and ADWIN stream telemetry.",
            action_label="Run Drift Analysis" if can_write() else None,
            action_fn=(
                lambda: (
                    safe(lambda: api.post("/analysis/run", json={"batch_ids": [bid]})),
                    st.rerun(),
                )
            )
            if can_write()
            else None,
            key=f"run_da_{bid}",
        )
        return

    with st.spinner(f"Loading drift analysis for {bid}…"):
        report = safe(lambda: api.get(f"/analysis/latest/{bid}"), retry_key=f"da_{bid}")
    if not report:
        return

    # 1. Summary strip
    _summary_row(report)

    # 2. Drift-types panel
    _drift_types_panel(report)

    # 3. Feature table with ProgressColumn for D, FI, S, DIS + agreement + severity
    section(
        "Feature-Level Drift Diagnostics",
        "Progress bars scale 0 → 1; hover column headers for metric definitions.",
    )
    rows = []
    for f in sorted(report["feature_details"], key=lambda x: x.get("dis") or 0.0, reverse=True):
        ag = f.get("detector_agreement") or {}
        sev = (f.get("severity") or "low").lower()
        sev_label = {"low": "Low", "moderate": "Moderate", "high": "High"}.get(sev, sev.title())
        rows.append(
            {
                "Feature": f["label"],
                "D": f.get("d") or 0.0,
                "FI": f.get("fi") or 0.0,
                "S": f.get("s") or 0.0,
                "DIS": f.get("dis") or 0.0,
                "PSI": f.get("psi"),
                "KS": f.get("ks_stat"),
                "KS p-value": fmt(f.get("ks_p"), ".2e", "—"),
                "JSD": f.get("jsd"),
                "Severity": sev_label,
                "Agreement": ag.get("status") or "—",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "D": st.column_config.ProgressColumn(
                "D", min_value=0.0, max_value=1.0, format="%.3f", help=theme.HELP_TEXTS["d"]
            ),
            "FI": st.column_config.ProgressColumn(
                "FI", min_value=0.0, max_value=1.0, format="%.3f", help=theme.HELP_TEXTS["fi"]
            ),
            "S": st.column_config.ProgressColumn(
                "S", min_value=0.0, max_value=1.0, format="%.3f", help=theme.HELP_TEXTS["s"]
            ),
            "DIS": st.column_config.ProgressColumn(
                "DIS", min_value=0.0, max_value=1.0, format="%.3f", help=theme.HELP_TEXTS["dis"]
            ),
            "PSI": st.column_config.NumberColumn("PSI", format="%.4f", help=theme.HELP_TEXTS["psi"]),
            "KS": st.column_config.NumberColumn("KS", format="%.4f", help=theme.HELP_TEXTS["ks"]),
            "KS p-value": st.column_config.TextColumn("KS p-value", help=theme.HELP_TEXTS["ks"]),
            "JSD": st.column_config.NumberColumn("JSD", format="%.4f", help=theme.HELP_TEXTS["jsd"]),
            "Severity": st.column_config.TextColumn("Severity"),
            "Agreement": st.column_config.TextColumn(
                "Detector Agreement",
                help="agree-drift: all detectors exceed noise floor; disagree: subset only; agree-stable: none.",
            ),
        },
    )

    # 4. Feature Detail Panel
    _feature_detail_panel(report)


def _drift_types_panel(report: dict[str, Any]) -> None:
    t = report.get("drift_types")
    if not t:
        return

    def _pill(v: Any) -> str:
        if v is None:
            return status_pill("Unknown – labels pending", "unknown", dot=False)
        return status_pill("Yes", "critical", dot=False) if v else status_pill("No", "low", dot=False)

    st.markdown(
        f"""
        <div class="dg-card" style="padding:14px 18px;margin-top:8px;">
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
                <div><div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:{theme.MUTED};margin-bottom:6px;">Data drift</div>{_pill(t.get('data_drift'))}</div>
                <div><div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:{theme.MUTED};margin-bottom:6px;" title="{theme.HELP_TEXTS['adwin']}">Prediction drift (ADWIN / KS) ⓘ</div>{_pill(t.get('prediction_drift'))}</div>
                <div><div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:{theme.MUTED};margin-bottom:6px;">Concept drift suspected</div>{_pill(t.get('concept_drift_suspected'))}</div>
                <div><div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;color:{theme.MUTED};margin-bottom:6px;">Confirmed degradation</div>{_pill(t.get('confirmed_degradation'))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
        info_callout(
            "warning",
            "<b>Evaluation notice:</b> The current champion was trained on part of "
            + ", ".join(report["leakage_warning"])
            + ". Performance on these batches is optimistic.",
        )


def _feature_detail_panel(report: dict[str, Any]) -> None:
    section(
        "Feature Detail Inspector",
        "Select any attribute to compare baseline vs production distributions and detector noise floors.",
    )
    feats = {f["label"]: f for f in report["feature_details"]}
    ranked = sorted(feats, key=lambda k: feats[k].get("dis") or 0.0, reverse=True)
    label = st.selectbox(
        "Inspect attribute", ranked, key=f"feat_detail_{report['job_id']}"
    )
    f = feats[label]
    left, right = st.columns([3, 2])
    with left:
        chart(
            distribution_figure(
                report["distributions"].get(f["feature"]),
                f"{label}: 2020 Baseline vs {report['batch_name']}",
                360,
                names={"baseline": "Baseline 2020", "production": report["batch_name"]},
            ),
            key="feat_dist",
        )
    with right:
        sev = (f.get("severity") or "low").lower()
        ag = (f.get("detector_agreement") or {}).get("status") or "—"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">'
            f'<span style="font-size:16px;font-weight:600;color:{theme.INK};">{label}</span>'
            f"{status_pill(f['drift_status'], sev)}"
            f"{status_pill(ag, 'neutral', dot=False)}"
            f"</div>",
            unsafe_allow_html=True,
        )
        det = pd.DataFrame(
            [
                {
                    "Detector": "PSI",
                    "Observed": f["psi"],
                    "Noise floor (q95)": (f["floors"] or {}).get("psi"),
                    "Normalised D": (f["normalised"] or {}).get("psi"),
                },
                {
                    "Detector": "KS statistic",
                    "Observed": f["ks_stat"],
                    "Noise floor (q95)": (f["floors"] or {}).get("ks"),
                    "Normalised D": (f["normalised"] or {}).get("ks"),
                },
                {
                    "Detector": "JSD",
                    "Observed": f["jsd"],
                    "Noise floor (q95)": (f["floors"] or {}).get("jsd"),
                    "Normalised D": (f["normalised"] or {}).get("jsd"),
                },
            ]
        )
        table(
            det,
            {"Observed": ".4f", "Noise floor (q95)": ".4f", "Normalised D": ".3f"},
            missing="n/a",
        )
        if f["ks_p"] is not None:
            st.caption(f"KS p-value: `{f['ks_p']:.2e}` · {theme.HELP_TEXTS['noise_floor']}")
        if f["type"] == "numerical" and f["reference_stats"]:
            rs, ps = f["reference_stats"], f["production_stats"]
            stats = pd.DataFrame({"Baseline": rs, "Production": ps}).loc[
                ["mean", "median", "std", "p05", "p95", "missing_rate"]
            ]
            stats.index = ["Mean", "Median", "Std Dev", "5th Pct", "95th Pct", "Missing Rate"]
            stats["Change"] = [
                (((p - b) / b if (b and p is not None) else None) if k != "Missing Rate" else None)
                for k, b, p in zip(stats.index, stats["Baseline"], stats["Production"], strict=True)
            ]
            table(
                stats.reset_index(names="Metric"),
                {"Baseline": ",.2f", "Production": ",.2f", "Change": "+.1%"},
            )
        elif f["type"] == "categorical":
            if f.get("new_categories"):
                info_callout("warning", "New categories in production: " + ", ".join(f["new_categories"]))
            if f.get("missing_categories"):
                info_callout("info", "Baseline categories absent: " + ", ".join(f["missing_categories"]))
