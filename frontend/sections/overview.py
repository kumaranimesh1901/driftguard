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
        "Banking Overview",
        "Credit-risk model surveillance: 2020 training baseline versus 2026 production customers",
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

    c = st.columns(6)
    kpi(
        c[0],
        "Customers monitored",
        f"{k['total_customers_monitored']:,}",
        f"vs {data['baseline']['row_count']:,} in baseline",
    )
    kpi(
        c[1],
        "Production batches",
        str(k["production_batches"]),
        f"{k['analysed_batches']} analysed",
    )
    kpi(
        c[2], "Features monitored", str(k["features_monitored"]), "financial attributes"
    )
    drifted = k["drifting_features"]
    sig = k["significant_drift_features"]
    kpi(
        c[3],
        "Drifted features",
        "—" if drifted is None else f"{drifted} / {k['features_monitored']}",
        (
            f"{sig} significant · {k['latest_batch_id']}"
            if k["latest_batch_id"]
            else "no analysis yet"
        ),
        accent=RISK["HIGH RISK"][0] if drifted else None,
    )
    risk = k["latest_risk_category"]
    kpi(
        c[4],
        "Overall DIS",
        fmt(k["overall_dis"]),
        "latest batch (proposed score)",
        accent=RISK.get(risk, (BLUE,))[0],
    )
    mh = k["model_health"]
    kpi(
        c[5],
        "Drift risk status",
        (risk or "Not analysed").replace(" RISK", ""),
        (
            f"model performance {mh['status']} ({mh['batch_id']})"
            if mh["confirmed"]
            else "model performance unconfirmed"
        ),
        accent=RISK.get(risk, (BLUE,))[0],
    )

    note(
        "<b>Drift is not degradation.</b> DIS ranks how consequential an input shift could be for the credit model. "
        "Model performance is only confirmed once loan outcome labels arrive for a batch."
    )

    trend = data["trend"]
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
            chart(layout(fig, 330, "DIS trend across production batches", legend=False))
        with right:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[t["batch_id"] for t in trend],
                    y=[
                        t["drifting_features_count"] - t["significant_drift_count"]
                        for t in trend
                    ],
                    name="Moderate drift",
                    marker_color="#E9A23B",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[t["batch_id"] for t in trend],
                    y=[t["significant_drift_count"] for t in trend],
                    name="Significant drift",
                    marker_color=RISK["CRITICAL RISK"][0],
                )
            )
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title="Features drifting", dtick=1)
            chart(layout(fig, 330, "Monthly drift trend (features drifting per batch)"))

        hm = data["heatmap"]
        fig = go.Figure(
            go.Heatmap(
                z=hm["d"],
                x=hm["batches"],
                y=hm["features"],
                zmin=0,
                zmax=1,
                colorscale=[
                    [0, "#F2F5F9"],
                    [0.15, "#DCE8F6"],
                    [0.5, "#5E93D1"],
                    [1, NAVY],
                ],
                text=[[fmt(v, ".2f") for v in row] for row in hm["d"]],
                texttemplate="%{text}",
                hovertemplate="%{y} · %{x}<br>D = %{z:.2f}<extra></extra>",
                colorbar={"title": "D"},
            )
        )
        fig.update_yaxes(autorange="reversed")
        chart(
            layout(
                fig,
                420,
                "Feature-wise drift severity (normalised drift D per batch)",
                legend=False,
            )
        )
    else:
        st.info(
            "Batches are registered but not analysed yet. Run analyses in **Dataset Management → Select & analyse**."
        )

    dists = data["distributions"]
    st.markdown("#### Historical versus production customer distributions")
    labels = {
        "monthly_income": "Monthly income",
        "credit_score": "Credit score",
        "outstanding_loan_amount": "Outstanding loan amount",
        "credit_limit": "Credit limit",
        "debt_to_income": "Debt-to-income",
        "fd_balance": "FD balance",
        "employment_type": "Employment type",
    }
    pick = st.segmented_control(
        "Attribute",
        list(labels),
        format_func=labels.get,
        default="debt_to_income",
        key="ov_attr",
    )
    if pick:
        chart(
            distribution_figure(
                dists.get(pick),
                f"{labels[pick]}: 2020 baseline vs each 2026 batch",
                340,
            ),
            key="ov_dist",
        )
    a, b, c3 = st.columns(3)
    with a:
        chart(
            distribution_figure(
                dists.get("monthly_income"),
                "Customer income distribution",
                290,
                legend=False,
            ),
            key="ov_inc",
        )
    with b:
        chart(
            distribution_figure(
                dists.get("credit_score"),
                "Credit score distribution",
                290,
                legend=False,
            ),
            key="ov_cs",
        )
    with c3:
        chart(
            distribution_figure(
                dists.get("outstanding_loan_amount"),
                "Credit exposure (outstanding loans)",
                290,
                legend=False,
            ),
            key="ov_exp",
        )
    st.caption(
        "Colours match the legend of the chart above (grey = 2020 baseline). Histogram ranges span the pooled "
        "0.5th–99.5th percentiles; values beyond are counted in the outermost bins."
    )

    rates = data["default_rates"]
    if rates:
        fig = go.Figure()
        names = [r["population"] for r in rates]
        actual = [r["actual_default_rate"] for r in rates]
        fig.add_trace(
            go.Bar(
                x=names,
                y=actual,
                name="Actual default rate (labelled)",
                marker_color=[BASELINE_COLOUR] + [NAVY] * (len(rates) - 1),
                text=[fmt(v, ".1%", "labels pending") for v in actual],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=names,
                y=[r["predicted_default_rate"] for r in rates],
                name="Model-predicted default rate",
                marker_color=SKY,
                text=[fmt(r["predicted_default_rate"], ".1%") for r in rates],
                textposition="outside",
            )
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", title="Default rate")
        chart(
            layout(
                fig,
                340,
                "Default rate comparison (actual where outcome labels exist, predicted by the 2020 model)",
            )
        )
        st.caption(
            "Batches without labels show only the predicted rate; actual default rates are never estimated."
        )


def _thresholds(data: dict) -> tuple[float, float]:
    """Batch DIS bands from the backend configuration."""
    t = data["thresholds"]
    return t["batch_dis_moderate"], t["batch_dis_high"]
