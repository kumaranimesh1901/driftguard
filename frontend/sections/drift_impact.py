"""D. Drift Impact Score: overall DIS, feature contributions, batch comparison and trend."""

import api_client as api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components import (
    NAVY,
    RISK,
    SKY,
    chart,
    header,
    layout,
    note,
    safe,
)

from sections.shared import _summary_row, select_analysis


def render() -> None:
    header(
        "Drift Impact Score (DIS)",
        "Proposed score prioritising drift by how much the credit model relies on and reacts to each feature",
    )
    note(
        "<b>DIS<sub>i</sub> = D<sub>i</sub> × FI<sub>i</sub> × S<sub>i</sub></b> — normalised drift × SHAP feature importance × "
        "permutation sensitivity of the 2020 credit model, each in [0, 1]. Overall DIS is the maximum feature DIS. "
        "DIS is a proposed research score, not an industry standard, and a high DIS signals potential impact — "
        "<b>not</b> confirmed loss of accuracy."
    )
    choice = select_analysis("dis_job")
    if not choice:
        return
    report = safe(lambda: api.get(f"/analysis/{choice['job_id']}"))
    if not report:
        return
    _summary_row(report)
    if report["dis_mode"] != "model":
        st.warning(
            "The baseline has no labelled outcomes, so no model is available: FI and S are set to 1 and DIS equals D."
        )

    feats = sorted(report["feature_details"], key=lambda f: f["dis"], reverse=True)
    labels = [f["label"] for f in feats]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[f["d"] for f in feats],
            name="D (normalised drift)",
            orientation="h",
            marker_color=SKY,
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[f["fi"] for f in feats],
            name="FI (SHAP importance)",
            orientation="h",
            marker_color="#2A9D8F",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[f["s"] for f in feats],
            name="S (sensitivity)",
            orientation="h",
            marker_color="#E9A23B",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[f["dis"] for f in feats],
            name="DIS",
            orientation="h",
            marker_color=NAVY,
        )
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 1.02])
    left, right = st.columns([3, 2])
    with left:
        chart(layout(fig, 620, "Feature DIS contributions"), key="dis_contrib")
    with right:
        st.markdown("##### Ablation variants")
        var = pd.DataFrame(
            [
                {
                    "Feature": f["label"],
                    "DIS": f["dis"],
                    "D only": f["variants"]["d_only"],
                    "D×FI": f["variants"]["d_fi"],
                    "Weighted sum": f["variants"]["weighted_sum"],
                    "Rank-based": f["variants"]["rank_based"],
                    "Severity": f["severity"],
                }
                for f in feats
            ]
        )
        st.dataframe(
            var.style.format(
                {
                    c: "{:.3f}"
                    for c in ["DIS", "D only", "D×FI", "Weighted sum", "Rank-based"]
                }
            ),
            hide_index=True,
            width="stretch",
            height=560,
        )
        st.caption(
            "A pure product is zero when any factor is zero — the variants show how rankings change without that property."
        )

    hist = safe(lambda: api.get("/analysis/history"), {}) or {}
    latest = hist.get("latest_per_batch", [])
    if latest:
        t = report["thresholds"]
        fig = go.Figure()
        fig.add_hrect(
            y0=t["batch_dis_high"],
            y1=1.05,
            fillcolor=RISK["HIGH RISK"][1],
            line_width=0,
            layer="below",
        )
        fig.add_hrect(
            y0=t["batch_dis_moderate"],
            y1=t["batch_dis_high"],
            fillcolor=RISK["MEDIUM RISK"][1],
            line_width=0,
            layer="below",
        )
        fig.add_trace(
            go.Bar(
                x=[r["batch_id"] for r in latest],
                y=[r["overall_dis"] for r in latest],
                marker_color=[RISK[r["risk_category"]][0] for r in latest],
                text=[
                    f"{r['overall_dis']:.2f}<br>{r['risk_category'].replace(' RISK', '')}"
                    for r in latest
                ],
                textposition="outside",
                name="Overall DIS",
            )
        )
        fig.update_yaxes(range=[0, 1.2], title="Overall DIS")
        left, right = st.columns(2)
        with left:
            chart(
                layout(
                    fig,
                    340,
                    f"Batch-wise DIS comparison (moderate ≥ {t['batch_dis_moderate']}, high ≥ {t['batch_dis_high']})",
                    legend=False,
                ),
                key="dis_hist",
            )
        with right:
            trend = go.Figure(
                go.Scatter(
                    x=[r["batch_date"] for r in latest],
                    y=[r["overall_dis"] for r in latest],
                    mode="lines+markers",
                    line={"color": NAVY, "width": 3},
                    marker={
                        "size": 11,
                        "color": [RISK[r["risk_category"]][0] for r in latest],
                    },
                    text=[r["batch_id"] for r in latest],
                    hovertemplate="%{text} (%{x})<br>DIS %{y:.3f}<extra></extra>",
                )
            )
            trend.add_hline(
                y=t["batch_dis_high"], line_dash="dot", line_color=RISK["HIGH RISK"][0]
            )
            trend.add_hline(
                y=t["batch_dis_moderate"],
                line_dash="dot",
                line_color=RISK["MEDIUM RISK"][0],
            )
            trend.update_yaxes(range=[0, 1.1], title="Overall DIS")
            trend.update_xaxes(title="Batch date")
            chart(
                layout(trend, 340, "Historical DIS trend over time", legend=False),
                key="dis_trend",
            )
