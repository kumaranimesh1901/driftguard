"""G. Model Health: baseline vs production performance once outcome labels exist."""

import api_client as api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components import (
    BASELINE_COLOUR,
    BLUE,
    NAVY,
    RISK,
    SKY,
    chart,
    fmt,
    header,
    layout,
    note,
    perf_badge,
    risk_badge,
    risk_cell,
    safe,
    table,
)

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1-score"),
    ("roc_auc", "ROC-AUC"),
]


def render() -> None:
    header(
        "Model Health",
        "Baseline credit-model performance versus production outcomes once loan default labels arrive",
    )
    data = safe(lambda: api.get("/model-health"))
    if not data:
        return
    if not data.get("has_baseline"):
        st.info("Register a baseline first.")
        return
    m = data["model"]["metrics"]
    if not m.get("has_model"):
        st.warning(m.get("reason", "No baseline model is available."))
        return
    note(
        "Statistical drift does <b>not</b> by itself mean the model's accuracy has dropped. Degradation is confirmed only "
        "when a batch has at least the minimum number of labelled outcomes and its AUC confidence interval falls below the baseline."
    )

    st.markdown("#### Baseline model (2020 holdout)")
    c = st.columns(6)
    for col, (k, lbl) in zip(c, METRICS, strict=False):
        col.metric(lbl, fmt(m.get(k), ".3f"))
    c[5].metric("Default rate", fmt(m.get("baseline_default_rate"), ".1%"))
    st.caption(
        f"{data['model']['algorithm']} · {m.get('evaluation')} · threshold {m.get('threshold'):.2f}"
    )

    batches = data["batches"]
    if not batches:
        st.info("No analysed batches yet.")
        return
    rows = []
    for b in batches:
        pm = b.get("production_metrics") or {}
        rows.append(
            {
                "Batch": b["batch_id"],
                "Date": b["batch_date"],
                "Labelled": b.get("n_labels", 0),
                "Performance": b["performance_state"],
                "ROC-AUC": pm.get("roc_auc"),
                "AUC 95% CI": (
                    f"[{pm['auc_ci'][0]:.3f}, {pm['auc_ci'][1]:.3f}]"
                    if pm.get("auc_ci")
                    else "—"
                ),
                "AUC change": (
                    -pm["auc_drop"] if pm.get("auc_drop") is not None else None
                ),
                "Predicted default rate": b.get("predicted_default_rate"),
                "Actual default rate": pm.get("actual_default_rate"),
                "Drift risk": b["risk_category"],
            }
        )
    st.markdown("#### Production evaluation by batch")
    table(
        pd.DataFrame(rows),
        {
            "ROC-AUC": ".3f",
            "AUC change": "+.3f",
            "Predicted default rate": ".1%",
            "Actual default rate": ".1%",
        },
        {"Drift risk": risk_cell},
        missing="pending",
    )

    ids = [b["batch_id"] for b in batches]
    labelled = [b["batch_id"] for b in batches if b.get("has_production_labels")]
    bid = st.selectbox(
        "Batch",
        ids,
        index=ids.index(labelled[-1]) if labelled else len(ids) - 1,
        key="mh_batch",
    )
    b = next(x for x in batches if x["batch_id"] == bid)
    st.markdown(
        f"**{bid}** &nbsp; {perf_badge(b['performance_state'])} &nbsp; drift: {risk_badge(b['risk_category'])}",
        unsafe_allow_html=True,
    )
    st.write(b["status_message"])
    pm = b.get("production_metrics")
    base_m = b["baseline_metrics"]

    if pm:
        left, right = st.columns([5, 4])
        with left:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[lbl for _, lbl in METRICS],
                    y=[base_m.get(k) for k, _ in METRICS],
                    name="Baseline 2020 (holdout)",
                    marker_color=BASELINE_COLOUR,
                    text=[fmt(base_m.get(k), ".3f") for k, _ in METRICS],
                    textposition="outside",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[lbl for _, lbl in METRICS],
                    y=[pm.get(k) for k, _ in METRICS],
                    name=f"{bid} (labelled)",
                    marker_color=NAVY,
                    text=[fmt(pm.get(k), ".3f") for k, _ in METRICS],
                    textposition="outside",
                )
            )
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 1.1])
            chart(layout(fig, 340, "Baseline vs production metrics"), key="mh_metrics")
        with right:
            a, bcol = st.columns(2)
            with a:
                chart(
                    _confusion(base_m["confusion_matrix"], "Baseline holdout"),
                    key="cm_base",
                )
            with bcol:
                chart(_confusion(pm["confusion_matrix"], bid), key="cm_prod")
    else:
        st.warning(
            f"Model performance degradation cannot yet be confirmed for {bid}. "
            "Recommendation: collect matured loan default outcomes for this batch and re-run the analysis."
        )

    left, right = st.columns(2)
    with left:
        h = b.get("probability_histogram")
        if h:
            centres = [
                (h["edges"][i] + h["edges"][i + 1]) / 2
                for i in range(len(h["edges"]) - 1)
            ]
            fig = go.Figure(
                go.Bar(
                    x=centres, y=h["proportions"], marker_color=BLUE, name="Customers"
                )
            )
            thr = base_m.get("threshold")
            fig.add_vline(
                x=thr,
                line_dash="dash",
                line_color=RISK["CRITICAL RISK"][0],
                annotation_text=f"decision threshold {thr:.2f}",
                annotation_position="top right",
                annotation_yshift=8,
            )
            fig.update_yaxes(tickformat=".0%", title="Share of customers")
            fig.update_xaxes(title="Predicted default probability", range=[0, 1])
            chart(
                layout(fig, 300, "Default prediction distribution", legend=False),
                key="mh_prob",
            )
    with right:
        names = ["Baseline 2020", bid]
        actual = [
            base_m.get("baseline_default_rate"),
            (pm or {}).get("actual_default_rate"),
        ]
        pred = [base_m.get("predicted_default_rate"), b.get("predicted_default_rate")]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=names,
                y=actual,
                name="Actual",
                marker_color=NAVY,
                text=[fmt(v, ".1%", "labels pending") for v in actual],
                textposition="outside",
            )
        )
        fig.add_trace(
            go.Bar(
                x=names,
                y=pred,
                name="Predicted",
                marker_color=SKY,
                text=[fmt(v, ".1%") for v in pred],
                textposition="outside",
            )
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        chart(layout(fig, 300, "Actual vs predicted default rate"), key="mh_rates")


def _confusion(cm: list[list[int]], title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Pred. repay", "Pred. default"],
            y=["Repaid", "Defaulted"],
            colorscale=[[0, "#F2F5F9"], [1, NAVY]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 14},
        )
    )
    fig.update_yaxes(autorange="reversed")
    return layout(fig, 300, title, legend=False)
