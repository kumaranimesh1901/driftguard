"""G. Model Health: champion card, AUC CI error-bar chart vs baseline, confusion matrices, empty state."""

import api_client as api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme
from components import (
    chart,
    empty_state,
    fmt,
    format_date,
    format_perf_state,
    format_timestamp,
    info_callout,
    kpi_card,
    layout,
    page_header,
    perf_badge,
    risk_badge,
    risk_cell,
    safe,
    section,
    status_pill,
    table,
)

from sections.shared import visible_batches

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1_score", "F1-score"),
    ("roc_auc", "ROC-AUC"),
]


def render() -> None:
    selected_meta = page_header(
        "Model Health",
        "Champion holdout metrics vs confirmed production performance once loan outcomes arrive",
        show_batch_selector=True,
    )
    with st.spinner("Loading model health…"):
        data = safe(lambda: api.get("/model-health"), retry_key="mh_data")
    if not data:
        return
    if not data.get("has_baseline"):
        empty_state(
            "NO BASELINE",
            "No baseline registered yet",
            "Register a labelled 2020 baseline in Dataset Management to train the champion credit model.",
        )
        return
    m = data["model"]["metrics"]
    if not m.get("has_model"):
        info_callout("warning", m.get("reason", "No baseline model is available."))
        return

    # 1. Champion Card
    models = safe(lambda: api.get("/models"), []) or []
    champ = next((x for x in models if x.get("status") == "champion"), None)
    ver_label = f"v{champ['version']}" if champ else f"v{m.get('version', 1)}"
    trained_at = format_timestamp((champ or data["model"]).get("trained_at"))
    st.markdown(
        f"""
        <div class="dg-card" style="border-left:4px solid {theme.PRIMARY};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:16px;font-weight:700;color:{theme.INK};">Champion Model {ver_label}</span>
                    {status_pill("Champion", "primary")}
                    <span style="font-size:13px;color:{theme.MUTED};">{data['model']['algorithm']}</span>
                </div>
                <div style="font-size:12px;color:{theme.MUTED};">
                    Trained {trained_at} · Threshold {fmt(m.get('threshold'), '.2f')}
                </div>
            </div>
            <div style="font-size:12px;color:{theme.MUTED};margin-top:4px;">{m.get('evaluation', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c = st.columns(6)
    for col, (k, lbl) in zip(c, METRICS, strict=False):
        kpi_card(col, lbl, fmt(m.get(k), ".3f"), "holdout", tone="primary")
    kpi_card(c[5], "Default Rate", fmt(m.get("baseline_default_rate"), ".1%"), "baseline", tone="neutral")

    info_callout(
        "info",
        "<b>Degradation requires labels.</b> Statistical drift does not prove accuracy loss. "
        "Performance degradation is confirmed only when labelled outcomes arrive and the production AUC 95% CI falls below baseline.",
    )

    batches = visible_batches(data["batches"])
    if not batches:
        empty_state("NO BATCHES", "No analysed batches yet", "Analyse production batches to compare model performance.")
        return

    # 2. Production AUC CI vs Baseline Line Chart (for all labelled batches)
    labelled_batches = [b for b in batches if b.get("has_production_labels") and (b.get("production_metrics") or {}).get("roc_auc") is not None]
    if labelled_batches:
        section(
            "Production ROC-AUC (95% CI) vs Baseline",
            "Error bars show bootstrap 95% confidence intervals against the baseline holdout ROC-AUC line.",
        )
        base_auc = m.get("roc_auc")
        fig = go.Figure()
        if base_auc is not None:
            fig.add_hline(
                y=base_auc,
                line_dash="dash",
                line_color=theme.BASELINE_COLOR,
                annotation_text=f"Baseline holdout AUC = {base_auc:.3f}",
                annotation_position="top right",
                annotation_font={"size": 11, "color": theme.MUTED},
            )
        xs = [b["batch_id"] for b in labelled_batches]
        ys = [b["production_metrics"]["roc_auc"] for b in labelled_batches]
        err_hi = [
            max(0.0, (b["production_metrics"]["auc_ci"][1] - y)) if b["production_metrics"].get("auc_ci") else 0.0
            for b, y in zip(labelled_batches, ys, strict=True)
        ]
        err_lo = [
            max(0.0, (y - b["production_metrics"]["auc_ci"][0])) if b["production_metrics"].get("auc_ci") else 0.0
            for b, y in zip(labelled_batches, ys, strict=True)
        ]
        colors = [
            theme.RISK_COLORS["CRITICAL RISK"][0]
            if b["performance_state"] == "severe"
            else (
                theme.RISK_COLORS["HIGH RISK"][0]
                if b["performance_state"] == "degraded"
                else theme.RISK_COLORS["LOW RISK"][0]
            )
            for b in labelled_batches
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker={"size": 12, "color": colors, "line": {"color": "white", "width": 2}},
                error_y={"type": "data", "symmetric": False, "array": err_hi, "arrayminus": err_lo, "color": theme.INK, "thickness": 2},
                text=[f"{y:.3f}" for y in ys],
                textposition="top right",
                cliponaxis=False,
                name="Production ROC-AUC",
            )
        )
        fig.update_yaxes(range=[0.0, 1.05], title="ROC-AUC")
        chart(layout(fig, 300, legend=False), key="mh_auc_ci")

    # 3. Batch Evaluation Table
    section("Production Evaluation by Batch")
    rows = []
    for b in batches:
        pm = b.get("production_metrics") or {}
        rows.append(
            {
                "Batch": b["batch_id"],
                "Date": format_date(b["batch_date"]),
                "Labelled": b.get("n_labels", 0),
                "Performance": format_perf_state(b["performance_state"]),
                "ROC-AUC": pm.get("roc_auc"),
                "AUC 95% CI": f"[{pm['auc_ci'][0]:.3f}, {pm['auc_ci'][1]:.3f}]" if pm.get("auc_ci") else "—",
                "AUC Change": (-pm["auc_drop"] if pm.get("auc_drop") is not None else None),
                "Predicted Default": b.get("predicted_default_rate"),
                "Actual Default": pm.get("actual_default_rate"),
                "Drift Risk": b["risk_category"],
            }
        )
    table(
        pd.DataFrame(rows),
        {"ROC-AUC": ".3f", "AUC Change": "+.3f", "Predicted Default": ".1%", "Actual Default": ".1%"},
        {"Drift Risk": risk_cell},
        missing="Unknown – labels pending",
    )

    # 4. Selected Batch Deep-Dive (synced with page_header)
    ids = [b["batch_id"] for b in batches]
    sel_bid = (selected_meta or {}).get("batch_id")
    bid = sel_bid if sel_bid in ids else ids[-1]
    b = next(x for x in batches if x["batch_id"] == bid)

    section(f"Batch Diagnostics · {bid}", b.get("status_message"))
    st.markdown(
        f"{perf_badge(b['performance_state'])} &nbsp; {risk_badge(b['risk_category'])}",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

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
                    name="Baseline (holdout)",
                    marker_color=theme.BASELINE_COLOR,
                    text=[fmt(base_m.get(k), ".3f") for k, _ in METRICS],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig.add_trace(
                go.Bar(
                    x=[lbl for _, lbl in METRICS],
                    y=[pm.get(k) for k, _ in METRICS],
                    name=f"{bid} (labelled)",
                    marker_color=theme.D_COLOR,
                    text=[fmt(pm.get(k), ".3f") for k, _ in METRICS],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 1.12], title="Score")
            chart(layout(fig, 340, "Baseline vs Production Metrics"), key="mh_metrics")
        with right:
            a, bcol = st.columns(2)
            with a:
                chart(_confusion(base_m["confusion_matrix"], "Baseline holdout"), key="cm_base")
            with bcol:
                chart(_confusion(pm["confusion_matrix"], bid), key="cm_prod")
    else:
        empty_state(
            "LABELS PENDING",
            "Performance can't be confirmed until loan outcomes arrive",
            f"Batch {bid} has {b.get('n_labels', 0)} labelled rows. Upload matured loan default outcomes and re-run analysis to compute ROC-AUC and confusion matrices.",
        )

    # 5. Probability Histogram + Default Rate Comparison
    left, right = st.columns(2)
    with left:
        h = b.get("probability_histogram")
        if h:
            centres = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["edges"]) - 1)]
            fig = go.Figure(go.Bar(x=centres, y=h["proportions"], marker_color=theme.D_COLOR, name="Customers"))
            thr = base_m.get("threshold")
            fig.add_vline(
                x=thr,
                line_dash="dash",
                line_color=theme.RISK_COLORS["CRITICAL RISK"][0],
                annotation_text=f"threshold {thr:.2f}",
                annotation_position="top right",
            )
            fig.update_yaxes(tickformat=".0%", title="Share of customers")
            fig.update_xaxes(title="Predicted default probability", range=[0, 1])
            chart(layout(fig, 300, "Predicted Default Probability Distribution", legend=False), key="mh_prob")
    with right:
        names = ["Baseline 2020", bid]
        actual = [base_m.get("baseline_default_rate"), (pm or {}).get("actual_default_rate")]
        pred = [base_m.get("predicted_default_rate"), b.get("predicted_default_rate")]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=names,
                y=actual,
                name="Actual",
                marker_color=theme.DIS_COLOR,
                text=[fmt(v, ".1%", "Unknown – labels pending") for v in actual],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.add_trace(
            go.Bar(
                x=names,
                y=pred,
                name="Predicted",
                marker_color=theme.D_COLOR,
                text=[fmt(v, ".1%") for v in pred],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero", title="Default rate")
        chart(layout(fig, 300, "Actual vs Predicted Default Rate"), key="mh_rates")


def _confusion(cm: list[list[int]], title: str) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=["Pred. repay", "Pred. default"],
            y=["Repaid", "Defaulted"],
            colorscale=[[0, "#F1F5F9"], [1, theme.DIS_COLOR]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm],
            texttemplate="%{text}",
            textfont={"size": 14},
        )
    )
    fig.update_yaxes(autorange="reversed")
    return layout(fig, 310, title, legend=False)
