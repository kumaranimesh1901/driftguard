"""I. Model Versions & Retraining: champion-challenger workflow (admin-approved)."""

import api_client as api
import pandas as pd
import streamlit as st
from components import fmt, header, note, safe


def _batches_with_labels() -> list[dict]:
    batches = safe(lambda: api.get("/batches"), []) or []
    return [b for b in batches if (b.get("labelled_rows") or 0) > 0]


def render() -> None:
    header(
        "Model Versions & Retraining",
        "Controlled champion–challenger retraining: labelled data only, independent "
        "post-drift test set, administrator approval, rollback",
    )
    note(
        "DriftGuard never retrains or deploys automatically. A challenger is trained only on "
        "<b>labelled</b> records (baseline training split + part of the labelled production "
        "records). The rest of the labelled production records form an independent post-drift "
        "test set on which champion and challenger are compared. Deployment is refused unless the "
        "AUC gain meets the configured minimum, its 95% CI excludes zero, and the challenger does not "
        "regress on the 2020 holdout."
    )

    models = safe(lambda: api.get("/models"), []) or []
    st.markdown("#### Model versions")
    if models:
        st.dataframe(
            pd.DataFrame(models)[
                ["version", "status", "algorithm", "roc_auc", "f1_score", "trained_at",
                 "training_batches", "evaluation"]
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No model registered yet.")

    is_admin = st.session_state.get("role") == "administrator"
    st.markdown("#### Train a challenger")
    labelled = _batches_with_labels()
    if not labelled:
        st.info("No production batch has outcome labels yet; retraining needs labelled data.")
    elif not is_admin:
        st.caption("Only administrators can start retraining.")
    else:
        options = {f"{b['batch_id']} ({b['labelled_rows']} labelled)": b["batch_id"]
                   for b in labelled}
        picked = st.multiselect("Labelled batches to learn from", list(options))
        if st.button("Train & evaluate challenger", disabled=not picked):
            with st.spinner("Training challenger and evaluating on the independent test set..."):
                out = safe(lambda: api.post("/models/retrain",
                                            json={"batch_ids": [options[p] for p in picked]}))
            if out:
                st.success(f"Challenger v{out['challenger_version']} created "
                           f"(event {out['event_id']}). Review it below.")
                st.rerun()

    st.markdown("#### Retraining events")
    events = safe(lambda: api.get("/models/retraining"), []) or []
    if not events:
        st.caption("No retraining events yet.")
    for ev in events:
        e = ev.get("evaluation") or {}
        gain = e.get("auc_gain") or {}
        title = (f"Event {ev['event_id']} · model {ev['old_model_id']} → {ev['new_model_id']} · "
                 f"{ev['decision'].upper()}")
        with st.expander(title, expanded=ev["decision"] == "pending"):
            c = st.columns(4)
            post = e.get("post_drift_test") or {}
            c[0].metric("Champion AUC (post-drift test)", fmt((post.get("champion") or {}).get("roc_auc")))
            c[1].metric("Challenger AUC (post-drift test)", fmt((post.get("challenger") or {}).get("roc_auc")))
            c[2].metric("AUC gain", fmt(gain.get("gain"), "+.3f"),
                        f"95% CI [{fmt(gain.get('ci_low'))}, {fmt(gain.get('ci_high'))}]")
            c[3].metric("Acceptable", "Yes" if e.get("acceptable") else "No")
            if e.get("criteria"):
                st.json(e["criteria"], expanded=False)
            if e.get("data"):
                st.caption(" · ".join(f"{k.replace('_', ' ')}: {v}" for k, v in e["data"].items()))
            if is_admin and ev["decision"] == "pending":
                a, b = st.columns(2)
                deploy = a.button("Approve & deploy", key=f"dep{ev['event_id']}",
                                  disabled=not e.get("acceptable"))
                if deploy and safe(lambda i=ev["event_id"]: api.post(f"/models/retraining/{i}/deploy")):
                    st.rerun()
                reject = b.button("Reject", key=f"rej{ev['event_id']}")
                if reject and safe(lambda i=ev["event_id"]: api.post(f"/models/retraining/{i}/reject")):
                    st.rerun()
            if (
                is_admin
                and ev["decision"] == "deployed"
                and st.button("Roll back to previous champion", key=f"rb{ev['event_id']}")
                and safe(lambda i=ev["event_id"]: api.post(f"/models/retraining/{i}/rollback"))
            ):
                st.rerun()
