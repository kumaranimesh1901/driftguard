"""D. Drift Impact Score: formula diagram, sorted DIS bar chart, top-feature explainer, ablations."""

import api_client as api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme
from components import (
    chart,
    empty_state,
    info_callout,
    layout,
    page_header,
    safe,
    section,
    status_pill,
)

from sections.shared import (
    _summary_row,
    hide_test_batches,
    is_test_batch,
    select_analysis,
)


def render() -> None:
    page_header(
        "Drift Impact Score (DIS)",
        "Prioritising statistical drift by model reliance (SHAP importance) and sensitivity",
        show_batch_selector=True,
    )

    # Formula card with inline diagram
    st.markdown(
        f"""
        <div class="dg-card" style="padding:16px 20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
                    <span style="background:{theme.D_COLOR}18;color:{theme.D_COLOR};border:1px solid {theme.D_COLOR}44;padding:6px 12px;border-radius:8px;font-weight:700;font-size:13px;" title="{theme.HELP_TEXTS['d']}">D · Normalised Drift</span>
                    <span style="color:{theme.MUTED};font-weight:700;">×</span>
                    <span style="background:{theme.FI_COLOR}18;color:{theme.FI_COLOR};border:1px solid {theme.FI_COLOR}44;padding:6px 12px;border-radius:8px;font-weight:700;font-size:13px;" title="{theme.HELP_TEXTS['fi']}">FI · SHAP Importance</span>
                    <span style="color:{theme.MUTED};font-weight:700;">×</span>
                    <span style="background:{theme.S_COLOR}18;color:{theme.S_COLOR};border:1px solid {theme.S_COLOR}44;padding:6px 12px;border-radius:8px;font-weight:700;font-size:13px;" title="{theme.HELP_TEXTS['s']}">S · Sensitivity</span>
                    <span style="color:{theme.MUTED};font-weight:700;">=</span>
                    <span style="background:{theme.DIS_COLOR};color:#FFFFFF;padding:6px 14px;border-radius:8px;font-weight:700;font-size:13px;" title="{theme.HELP_TEXTS['dis']}">DIS ∈ [0, 1]</span>
                </div>
                <div style="font-size:12px;color:{theme.MUTED};max-width:420px;">
                    Overall batch DIS = max<sub>i</sub>(DIS<sub>i</sub>). Signals <b>potential</b> prediction impact — not confirmed accuracy loss.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    choice = select_analysis("dis_job")
    if not choice:
        return
    with st.spinner("Loading DIS breakdown…"):
        report = safe(lambda: api.get(f"/analysis/{choice['job_id']}"), retry_key="dis_rep")
    if not report:
        return
    _summary_row(report)

    if report["dis_mode"] != "model":
        info_callout(
            "warning",
            "The baseline has no labelled outcomes, so no model is available: FI and S default to 1.0 and DIS equals D.",
        )

    feats = sorted(report["feature_details"], key=lambda f: f["dis"], reverse=True)
    if not feats:
        empty_state("NO FEATURES", "No feature DIS scores available", "Re-run the analysis on a valid batch.")
        return

    # "Why is this feature first?" box using API evidence
    top = feats[0]
    ev_lines = report.get("evidence") or []
    ev_html = "".join(f"<li style='margin:2px 0;'>{line}</li>" for line in ev_lines[:3])
    st.markdown(
        f"""
        <div class="dg-card" style="border-left:4px solid {theme.PRIMARY};padding:14px 18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:{theme.PRIMARY};">Why is {top['label']} ranked #1?</div>
                {status_pill(f"DIS {top['dis']:.3f}", top.get('severity', 'low'))}
            </div>
            <p style="font-size:13px;color:{theme.TEXT};margin:6px 0 6px 0;">
                <b>{top['label']}</b> leads this batch because its normalised drift is <b>D = {top['d']:.3f}</b>,
                the credit model relies on it with SHAP importance <b>FI = {top['fi']:.3f}</b>, and permutation sensitivity is <b>S = {top['s']:.3f}</b>.
            </p>
            {"<ul style='margin:0;padding-left:18px;font-size:12px;color:" + theme.MUTED + ";'>" + ev_html + "</ul>" if ev_html else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        section("DIS by Feature (Annotated with D, FI, S)", "Sorted highest to lowest impact")
        labels = [f["label"] for f in feats]
        fig = go.Figure(
            go.Bar(
                y=labels,
                x=[f["dis"] for f in feats],
                orientation="h",
                marker_color=theme.DIS_COLOR,
                text=[
                    f"<b>{f['dis']:.3f}</b>  (D {f['d']:.2f} · FI {f['fi']:.2f} · S {f['s']:.2f})"
                    for f in feats
                ],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>DIS %{x:.3f}<extra></extra>",
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, 1.32], title="Drift Impact Score (DIS)")
        chart(layout(fig, 520, legend=False), key="dis_sorted_bar")

        with st.expander("View D / FI / S component breakdown chart", expanded=False):
            comp = go.Figure()
            comp.add_trace(go.Bar(y=labels, x=[f["d"] for f in feats], name="D (drift)", orientation="h", marker_color=theme.D_COLOR))
            comp.add_trace(go.Bar(y=labels, x=[f["fi"] for f in feats], name="FI (importance)", orientation="h", marker_color=theme.FI_COLOR))
            comp.add_trace(go.Bar(y=labels, x=[f["s"] for f in feats], name="S (sensitivity)", orientation="h", marker_color=theme.S_COLOR))
            comp.add_trace(go.Bar(y=labels, x=[f["dis"] for f in feats], name="DIS", orientation="h", marker_color=theme.DIS_COLOR))
            comp.update_layout(barmode="group", bargap=0.25)
            comp.update_yaxes(autorange="reversed")
            comp.update_xaxes(range=[0, 1.05], title="Score [0, 1]")
            chart(layout(comp, 540, "Component Decomposition: D, FI, S and DIS"), key="dis_contrib")

    with right:
        section("Ablation Comparison", "How feature rankings change across score formulations")
        var = pd.DataFrame(
            [
                {
                    "Feature": f["label"],
                    "DIS": f["dis"],
                    "D only": f["variants"]["d_only"],
                    "D×FI": f["variants"]["d_fi"],
                    "Weighted": f["variants"]["weighted_sum"],
                    "Rank": f["variants"]["rank_based"],
                }
                for f in feats
            ]
        )
        st.dataframe(
            var,
            hide_index=True,
            width="stretch",
            height=450,
            column_config={
                "DIS": st.column_config.NumberColumn("DIS", format="%.3f", help=theme.HELP_TEXTS["dis"]),
                "D only": st.column_config.NumberColumn("D only", format="%.3f"),
                "D×FI": st.column_config.NumberColumn("D×FI", format="%.3f"),
                "Weighted": st.column_config.NumberColumn("Weighted", format="%.3f"),
                "Rank": st.column_config.NumberColumn("Rank", format="%.3f"),
            },
        )
        info_callout(
            "info",
            "<b>Experiment finding:</b> In our counterfactual prioritisation experiment "
            "(<code>docs/experiments/dis_prioritisation_results.md</code>), <b>D×FI</b> performed as well as DIS "
            "(NDCG@3 0.932 vs 0.910); permutation sensitivity <b>S</b> adds little because it correlates strongly with FI.",
        )

    hist = safe(lambda: api.get("/analysis/history"), {}) or {}
    latest = [
        r
        for r in hist.get("latest_per_batch", [])
        if not (hide_test_batches() and is_test_batch(r.get("batch_id")))
    ]
    if latest:
        t = report["thresholds"]
        section("Batch-Wise DIS Comparison", f"Moderate ≥ {t['batch_dis_moderate']:.2f} · High ≥ {t['batch_dis_high']:.2f}")
        fig = go.Figure()
        fig.add_hrect(y0=t["batch_dis_high"], y1=1.05, fillcolor=theme.RISK_COLORS["HIGH RISK"][1], line_width=0, layer="below")
        fig.add_hrect(y0=t["batch_dis_moderate"], y1=t["batch_dis_high"], fillcolor=theme.RISK_COLORS["MEDIUM RISK"][1], line_width=0, layer="below")
        fig.add_trace(
            go.Bar(
                x=[r["batch_id"] for r in latest],
                y=[r["overall_dis"] for r in latest],
                marker_color=[theme.risk_color(r["risk_category"])[0] for r in latest],
                text=[f"{r['overall_dis']:.3f}" for r in latest],
                textposition="outside",
                cliponaxis=False,
                name="Overall DIS",
            )
        )
        fig.update_yaxes(range=[0, 1.12], title="Overall DIS")
        chart(layout(fig, 320, legend=False), key="dis_hist")
