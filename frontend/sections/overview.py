"""A. Overview: KPIs and trends for all production batches."""

import api_client as api
import plotly.graph_objects as go
import streamlit as st
from components import (
    BASELINE_COLOUR,
    BLUE,
    NAVY,
    RISK,
    SKY,
    chart,
    distribution_figure,
    fmt,
    header,
    kpi,
    layout,
    note,
    safe,
    synthetic_notice,
)

from sections.shared import is_admin


def render() -> None:
    header(
        "Banking Surveillance Overview",
        "Continuous surveillance of loan applicants: 2020 training baseline vs 2026 production batches",
    )
    data = safe(lambda: api.get("/overview"))
    if data is None:
        return
    if not data["baseline"]:
        st.info(
            "No historical baseline is registered yet. Go to **Dataset Management** to upload the 2020 baseline"
            + (", or use **Demo data (synthetic)** there." if is_admin() else ".")
        )
        return

    synthetic_notice(data["any_synthetic"])
    k = data["kpis"]
    thresholds = _thresholds(data)

    # Top Posture Banner
    risk = k["latest_risk_category"] or "LOW RISK"
    accent_fg, accent_bg = RISK.get(risk, (BLUE, "#EFF6FF"))
    mh = k["model_health"]

    st.markdown(
        f"""
        <div style="background:{accent_bg}; border:1px solid {accent_fg}44; border-radius:12px; padding:14px 20px; margin-bottom:18px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:{accent_fg};">Portfolio Health Status</span>
                <div style="font-size:1.15rem; font-weight:800; color:{accent_fg}; margin-top:2px;">{risk} on Latest Batch ({k['latest_batch_id'] or 'N/A'})</div>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.8rem; font-weight:600; color:{accent_fg};">Model State: <b>{mh['status'].upper()}</b> ({'Labels Confirmed' if mh['confirmed'] else 'Pending Labels'})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 6 KPIs
    c = st.columns(6)
    kpi(
        c[0],
        "Customers Monitored",
        f"{k['total_customers_monitored']:,}",
        f"vs {data['baseline']['row_count']:,} in baseline",
        accent=BLUE,
    )
    kpi(
        c[1],
        "Batches Monitored",
        str(k["production_batches"]),
        f"{k['analysed_batches']} fully analysed",
        accent=BLUE,
    )
    kpi(
        c[2],
        "Attributes Monitored",
        str(k["features_monitored"]),
        "financial features",
        accent=BLUE,
    )
    drifted = k["drifting_features"]
    sig = k["significant_drift_features"]
    kpi(
        c[3],
        "Drifted Features",
        "—" if drifted is None else f"{drifted} / {k['features_monitored']}",
        f"{sig} significant" if sig else "stable",
        accent=RISK["HIGH RISK"][0] if drifted else "#15803D",
    )
    kpi(
        c[4],
        "Overall DIS",
        fmt(k["overall_dis"]),
        "latest batch impact",
        accent=RISK.get(risk, (BLUE,))[0],
    )
    kpi(
        c[5],
        "Risk Category",
        risk.replace(" RISK", ""),
        "decision engine status",
        accent=RISK.get(risk, (BLUE,))[0],
    )

    note(
        "<b>Drift is not degradation.</b> Drift Impact Score (DIS) ranks how consequential an input shift could be "
        "for credit underwriting. Model performance degradation is only confirmed once actual loan outcomes arrive."
    )

    # Tabbed Views for Clean Navigation
    tab_trends, tab_heatmap, tab_dists, tab_performance = st.tabs(
        [
            "📈 Drift Trends & Alerts",
            "🔥 Severity Heatmap",
            "📊 Feature Distributions",
            "⚖️ Default Rate Surveillance",
        ]
    )

    trend = data["trend"]

    with tab_trends:
        if trend:
            left, right = st.columns(2)
            with left:
                fig = go.Figure()
                fig.add_hrect(
                    y0=thresholds[1],
                    y1=1.05,
                    fillcolor=RISK["HIGH RISK"][1],
                    line_width=0,
                    layer="below",
                )
                fig.add_hrect(
                    y0=thresholds[0],
                    y1=thresholds[1],
                    fillcolor=RISK["MEDIUM RISK"][1],
                    line_width=0,
                    layer="below",
                )
                fig.add_trace(
                    go.Scatter(
                        x=[t["batch_id"] for t in trend],
                        y=[t["overall_dis"] for t in trend],
                        mode="lines+markers+text",
                        line={"color": NAVY, "width": 3},
                        text=[fmt(t["overall_dis"], ".2f") for t in trend],
                        textposition="top center",
                        name="Overall DIS",
                        marker={
                            "size": 12,
                            "color": [RISK[t["risk_category"]][0] for t in trend],
                            "line": {"color": "white", "width": 2},
                        },
                        customdata=[[t["risk_category"], t["batch_date"]] for t in trend],
                        hovertemplate="%{x} (%{customdata[1]})<br>DIS %{y:.3f}<br>%{customdata[0]}<extra></extra>",
                    )
                )
                fig.update_yaxes(range=[0, 1.1], title="Overall DIS (max feature DIS)")
                chart(layout(fig, 340, "DIS Trend Across Production Batches", legend=False))
            with right:
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=[t["batch_id"] for t in trend],
                        y=[
                            t["drifting_features_count"] - t["significant_drift_count"]
                            for t in trend
                        ],
                        name="Moderate Drift",
                        marker_color="#E9A23B",
                    )
                )
                fig.add_trace(
                    go.Bar(
                        x=[t["batch_id"] for t in trend],
                        y=[t["significant_drift_count"] for t in trend],
                        name="Significant Drift",
                        marker_color=RISK["CRITICAL RISK"][0],
                    )
                )
                fig.update_layout(barmode="stack")
                fig.update_yaxes(title="Features Drifting Count", dtick=1)
                chart(layout(fig, 340, "Monthly Drift Trend (Features Drifting per Batch)"))
        else:
            st.info("No analyses yet. Upload and analyse production batches to see trends.")

    with tab_heatmap:
        if trend:
            hm = data["heatmap"]
            fig = go.Figure(
                go.Heatmap(
                    z=hm["d"],
                    x=hm["batches"],
                    y=hm["features"],
                    zmin=0,
                    zmax=1,
                    colorscale=[
                        [0, "#F8FAFC"],
                        [0.15, "#DBEAFE"],
                        [0.5, "#60A5FA"],
                        [1, NAVY],
                    ],
                    text=[[fmt(v, ".2f") for v in row] for row in hm["d"]],
                    texttemplate="%{text}",
                    hovertemplate="%{y} · %{x}<br>Normalized Drift D = %{z:.2f}<extra></extra>",
                    colorbar={"title": "D"},
                )
            )
            fig.update_yaxes(autorange="reversed")
            chart(
                layout(
                    fig,
                    440,
                    "Feature-wise Normalized Drift Severity (D) Matrix",
                    legend=False,
                )
            )
        else:
            st.info("Heatmap requires at least one analysed batch.")

    with tab_dists:
        dists = data["distributions"]
        st.markdown("##### Compare Historical vs Production Customer Distributions")
        labels = {
            "debt_to_income": "Debt-to-Income (DTI)",
            "monthly_income": "Monthly Income",
            "credit_score": "Credit Score",
            "outstanding_loan_amount": "Outstanding Loans",
            "credit_limit": "Credit Limit",
            "fd_balance": "FD Balance",
            "employment_type": "Employment Type",
        }
        pick = st.selectbox(
            "Select Financial Attribute to Inspect",
            list(labels),
            format_func=labels.get,
            key="ov_attr",
        )
        if pick:
            chart(
                distribution_figure(
                    dists.get(pick),
                    f"{labels[pick]}: 2020 Baseline vs 2026 Production Batches",
                    350,
                ),
                key="ov_dist",
            )

        st.markdown("##### Core Risk Profiles Summary")
        a, b, c3 = st.columns(3)
        with a:
            chart(
                distribution_figure(
                    dists.get("monthly_income"),
                    "Monthly Income",
                    260,
                    legend=False,
                ),
                key="ov_inc",
            )
        with b:
            chart(
                distribution_figure(
                    dists.get("credit_score"),
                    "Credit Score",
                    260,
                    legend=False,
                ),
                key="ov_cs",
            )
        with c3:
            chart(
                distribution_figure(
                    dists.get("outstanding_loan_amount"),
                    "Outstanding Debt",
                    260,
                    legend=False,
                ),
                key="ov_exp",
            )

    with tab_performance:
        rates = data["default_rates"]
        if rates:
            fig = go.Figure()
            names = [r["population"] for r in rates]
            actual = [r["actual_default_rate"] for r in rates]
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=actual,
                    name="Actual Default Rate (Labelled Outcomes)",
                    marker_color=[BASELINE_COLOUR] + [NAVY] * (len(rates) - 1),
                    text=[fmt(v, ".1%", "Pending labels") for v in actual],
                    textposition="outside",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=[r["predicted_default_rate"] for r in rates],
                    name="Model-Predicted Default Rate",
                    marker_color=SKY,
                    text=[fmt(r["predicted_default_rate"], ".1%") for r in rates],
                    textposition="outside",
                )
            )
            fig.update_layout(barmode="group")
            fig.update_yaxes(tickformat=".0%", title="Default Rate")
            chart(
                layout(
                    fig,
                    350,
                    "Default Rate Comparison: Actual vs 2020 Model Predicted",
                )
            )
            st.caption(
                "Batches without outcome labels display only the model's predicted default rate; actual defaults are only shown once confirmed."
            )
        else:
            st.info("No default rate data available.")


def _thresholds(data: dict) -> tuple[float, float]:
    """Batch DIS bands from the backend configuration."""
    t = data["thresholds"]
    return t["batch_dis_moderate"], t["batch_dis_high"]
