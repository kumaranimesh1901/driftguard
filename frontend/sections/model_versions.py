"""I. Model Versions & Retraining: version timeline, comparison cards, CI chart & st.dialog confirmations."""

import api_client as api
import plotly.graph_objects as go
import streamlit as st
import theme
from components import (
    chart,
    empty_state,
    fmt,
    format_timestamp,
    info_callout,
    layout,
    page_header,
    safe,
    section,
    status_pill,
)

from sections.shared import visible_batches

STATUS_TONE = {
    "champion": "primary",
    "challenger": "warning",
    "retired": "neutral",
    "rolled_back": "neutral",
    "deployed": "low",
    "pending": "warning",
    "rejected": "danger",
}


def _batches_with_labels() -> list[dict]:
    batches = visible_batches(safe(lambda: api.get("/batches"), []) or [])
    return [b for b in batches if (b.get("labelled_rows") or 0) > 0]


@st.dialog("Confirm Model Action")
def _confirm_dialog(action: str, event_id: int, summary: str) -> None:
    st.markdown(f"**{action.title()} retraining event #{event_id}?**")
    st.caption(summary)
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    btn_type = "primary" if action == "deploy" else "secondary"
    if c2.button(f"Confirm {action}", type=btn_type, width="stretch"):
        out = safe(lambda: api.post(f"/models/retraining/{event_id}/{action}"))
        if out:
            st.rerun()


def render() -> None:
    page_header(
        "Model Versions & Retraining",
        "Controlled champion–challenger workflow: labelled data only, independent post-drift test set, admin approval & rollback",
        show_batch_selector=True,
    )
    info_callout(
        "info",
        "<b>Human-in-the-loop governance:</b> DriftGuard never retrains or deploys automatically. "
        "Challengers train on labelled rows only and are evaluated on a 40% stratified post-drift test split; "
        "deployment is refused unless AUC gain meets the minimum, 95% CI > 0, and baseline holdout does not regress.",
    )

    models = safe(lambda: api.get("/models"), [], retry_key="mv_models") or []
    events = safe(lambda: api.get("/models/retraining"), [], retry_key="mv_events") or []

    # 1. Version Timeline
    section("Model Version Timeline", "Chronological progression from initial baseline model to current champion")
    if not models:
        empty_state("NO MODELS", "No model registered yet", "Register a labelled baseline in Dataset Management.")
    else:
        cards_html = []
        for idx, m in enumerate(models):
            arrow = (
                f'<div style="display:flex;align-items:center;color:{theme.MUTED};font-size:18px;font-weight:700;">→</div>'
                if idx > 0
                else ""
            )
            tone = STATUS_TONE.get(m["status"], "neutral")
            border_c = theme.PRIMARY if m["status"] == "champion" else theme.BORDER
            cards_html.append(
                f"{arrow}"
                f'<div style="background:{theme.SURFACE};border:1.5px solid {border_c};border-radius:10px;padding:12px 14px;min-width:170px;flex:1;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">'
                f'<span style="font-weight:700;font-size:15px;color:{theme.INK};">v{m["version"]}</span>'
                f'{status_pill(m["status"].title(), tone)}'
                f"</div>"
                f'<div style="font-size:13px;font-weight:600;color:{theme.TEXT};margin-top:8px;font-variant-numeric:tabular-nums;">AUC {fmt(m.get("roc_auc"))} · F1 {fmt(m.get("f1_score"))}</div>'
                f'<div style="font-size:11px;color:{theme.MUTED};margin-top:3px;">{format_timestamp(m.get("trained_at"))}</div>'
                f"</div>"
            )
        st.markdown(
            f'<div style="display:flex;gap:10px;overflow-x:auto;padding-bottom:6px;margin-bottom:14px;">{"".join(cards_html)}</div>',
            unsafe_allow_html=True,
        )

    # 2. Train a Challenger
    is_admin = st.session_state.get("role") == "administrator"
    section("Train a Challenger", "Select labelled production batches to train and evaluate a new candidate")
    labelled = _batches_with_labels()
    if not labelled:
        empty_state(
            "NO LABELS",
            "No labelled production batches available",
            "Retraining requires labelled production outcomes (unlabelled rows are never used).",
        )
    elif not is_admin:
        info_callout("info", "Only administrators can initiate retraining, deploy challengers, or roll back models.")
    else:
        options = {f"{b['batch_id']} ({b['labelled_rows']:,} labelled)": b["batch_id"] for b in labelled}
        c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
        picked = c1.multiselect("Labelled batches", list(options), key="mv_pick_batches")
        if c2.button("Train & evaluate challenger", type="primary", disabled=not picked, width="stretch"):
            with st.spinner("Training challenger and evaluating on independent post-drift test set…"):
                out = safe(
                    lambda: api.post(
                        "/models/retrain",
                        json={"batch_ids": [options[p] for p in picked]},
                    )
                )
            if out:
                st.success(
                    f"Challenger v{out['challenger_version']} created (event #{out['event_id']}). Review it below."
                )
                st.rerun()

    # 3. Retraining Events Comparison Cards
    section("Retraining Events", "Champion vs challenger evaluation on the independent 40% post-drift test split")
    if not events:
        empty_state("NO EVENTS", "No retraining events yet", "Train a challenger above to compare against the champion.")
        return

    for ev in events:
        e = ev.get("evaluation") or {}
        gain = e.get("auc_gain") or {}
        post = e.get("post_drift_test") or {}
        crit = e.get("criteria") or {}
        decision = ev["decision"]
        tone = STATUS_TONE.get(decision, "neutral")
        title = f"Event #{ev['event_id']} · model #{ev['old_model_id']} → #{ev['new_model_id']} · {decision.upper()}"

        with st.expander(title, expanded=(decision in ("pending", "deployed"))):
            c_left, c_right = st.columns([1.3, 1])
            with c_left:
                m1, m2, m3 = st.columns(3)
                m1.metric("Champion AUC", fmt((post.get("champion") or {}).get("roc_auc")))
                m2.metric("Challenger AUC", fmt((post.get("challenger") or {}).get("roc_auc")))
                m3.metric(
                    "AUC Gain",
                    fmt(gain.get("gain"), "+.3f"),
                    f"95% CI [{fmt(gain.get('ci_low'))}, {fmt(gain.get('ci_high'))}]",
                )

                # AUC gain + 95% CI horizontal bar
                if gain.get("gain") is not None and gain.get("ci_low") is not None:
                    fig = go.Figure()
                    min_g = crit.get("min_gain_auc", 0.01)
                    fig.add_vline(x=0, line_color=theme.MUTED, line_width=1.5)
                    fig.add_vline(
                        x=min_g,
                        line_dash="dash",
                        line_color=theme.PRIMARY,
                        annotation_text=f"min gain +{min_g:.2f}",
                        annotation_position="top right",
                    )
                    bar_color = theme.RISK_COLORS["LOW RISK"][0] if e.get("acceptable") else theme.RISK_COLORS["CRITICAL RISK"][0]
                    fig.add_trace(
                        go.Scatter(
                            x=[gain["gain"]],
                            y=["AUC gain"],
                            mode="markers",
                            marker={"size": 12, "color": bar_color},
                            error_x={
                                "type": "data",
                                "symmetric": False,
                                "array": [max(0.0, gain["ci_high"] - gain["gain"])],
                                "arrayminus": [max(0.0, gain["gain"] - gain["ci_low"])],
                                "color": theme.INK,
                                "thickness": 2,
                            },
                            hovertemplate="Gain %{x:+.3f}<extra></extra>",
                        )
                    )
                    fig.update_xaxes(title="Paired bootstrap AUC gain (95% CI)")
                    chart(layout(fig, 150, legend=False), key=f"ci_ev_{ev['event_id']}")

            with c_right:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                    f'<span style="font-size:12px;font-weight:700;text-transform:uppercase;color:{theme.MUTED};">Acceptance Criteria</span>'
                    f'{status_pill(decision.replace("_", " ").title(), tone)}'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                checks = [
                    ("Gain meets minimum", crit.get("gain_meets_minimum")),
                    ("95% CI excludes zero", crit.get("gain_ci_excludes_zero")),
                    ("No baseline regression", crit.get("no_baseline_regression")),
                ]
                chk_html = "".join(
                    f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #F1F5F9;font-size:13px;">'
                    f'<span>{lbl}</span>'
                    f'<span style="font-weight:700;color:{theme.RISK_COLORS["LOW RISK"][0] if ok else theme.RISK_COLORS["CRITICAL RISK"][0]};">'
                    f'{"✓ Pass" if ok else "✗ Fail"}</span></div>'
                    for lbl, ok in checks
                )
                st.markdown(chk_html, unsafe_allow_html=True)
                if e.get("data"):
                    d = e["data"]
                    st.caption(
                        f"Train: {d.get('baseline_train_rows', 0):,} baseline + {d.get('production_train_rows', 0):,} prod · "
                        f"Post-drift test: {d.get('post_drift_test_rows', 0):,} · "
                        f"Unlabelled used: {d.get('unlabelled_rows_used', 0)}"
                    )

            # Action buttons with st.dialog confirmation
            if is_admin and decision == "pending":
                b1, b2, _ = st.columns([1.2, 1, 2.5])
                if b1.button(
                    "Approve & deploy",
                    key=f"dep_{ev['event_id']}",
                    type="primary",
                    disabled=not e.get("acceptable"),
                    width="stretch",
                ):
                    _confirm_dialog(
                        "deploy",
                        ev["event_id"],
                        f"This will retire model #{ev['old_model_id']} and promote model #{ev['new_model_id']} to champion in one transaction.",
                    )
                if b2.button("Reject", key=f"rej_{ev['event_id']}", type="secondary", width="stretch"):
                    _confirm_dialog(
                        "reject",
                        ev["event_id"],
                        f"This will retire challenger model #{ev['new_model_id']} without deploying it.",
                    )
            elif is_admin and decision == "deployed":
                b1, _ = st.columns([1.5, 3])
                if b1.button(
                    "Roll back to previous champion",
                    key=f"rb_{ev['event_id']}",
                    type="secondary",
                    width="stretch",
                ):
                    _confirm_dialog(
                        "rollback",
                        ev["event_id"],
                        f"This will retire model #{ev['new_model_id']} and restore model #{ev['old_model_id']} as champion.",
                    )
