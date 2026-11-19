"""E. Batch Monitoring: batch list, status, individual and combined comparisons."""

from datetime import date

import api_client as api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from components import (
    RISK,
    chart,
    distribution_figure,
    fmt,
    header,
    layout,
    risk_badge,
    risk_cell,
    safe,
    status_cell,
    table,
)

from sections.shared import FEATURE_LABELS, can_write, feature_table

BASELINE = "Baseline 2020"


def render() -> None:
    header(
        "Batch Monitoring",
        "Track drift across production batches and compare any two populations",
    )
    batches = safe(lambda: api.get("/batches"), []) or []
    if not batches:
        st.info("Upload production batches first.")
        return
    hist = safe(lambda: api.get("/analysis/history"), {}) or {}

    f1, f2, f3 = st.columns([1, 1, 2])
    dates = [date.fromisoformat(b["batch_date"]) for b in batches]
    start = f1.date_input("From", min(dates), key="cmp_from")
    end = f2.date_input("To", max(dates), key="cmp_to")
    risks = f3.multiselect(
        "Risk category", list(RISK), default=list(RISK), key="cmp_risk"
    )
    rows = []
    for b in batches:
        la = b.get("latest_analysis") or {}
        d = date.fromisoformat(b["batch_date"])
        if not (start <= d <= end):
            continue
        if la and la.get("risk_category") not in risks:
            continue
        rows.append(
            {
                "Batch": b["batch_id"],
                "Batch date": b["batch_date"],
                "Customers": b["row_count"],
                "Overall DIS": la.get("overall_dis"),
                "Status": la.get("risk_category", "Not analysed"),
                "Drifting features": la.get("drifting_features_count"),
                "Performance": la.get("performance_state"),
                "Analysed at": la.get("analysed_at"),
                "Uploaded at": b["uploaded_at"],
                "Source": "Synthetic" if b.get("is_synthetic") else "Uploaded",
            }
        )
    listing = pd.DataFrame(rows)
    if listing.empty:
        st.info("No batches match the filters.")
    else:
        table(listing, {"Overall DIS": ".3f"}, {"Status": risk_cell})

    fd = hist.get("first_drift") or {}
    if fd:
        c1, c2 = st.columns(2)
        first = fd.get("first_moderate_or_higher")
        high = fd.get("first_high")
        c1.markdown(
            f'<div class="dg-card"><h4>When drift first appeared</h4>'
            f'{"First moderate-or-higher drift: <b>" + first["batch_id"] + "</b> (" + first["batch_date"] + ")" if first else "No drift above the low band yet."}<br>'
            f'{"First high drift: <b>" + high["batch_id"] + "</b> (" + high["batch_date"] + ")" if high else "No high drift yet."}</div>',
            unsafe_allow_html=True,
        )
        per = fd.get("first_significant_by_feature") or {}
        c2.markdown(
            '<div class="dg-card"><h4>First significant drift by feature</h4>'
            + (
                "<br>".join(
                    f"{FEATURE_LABELS.get(k, k)}: <b>{v}</b>" for k, v in per.items()
                )
                or "None yet."
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    runs = hist.get("runs", [])
    if runs:
        individual = [r for r in runs if r["analysis_type"] == "individual"]
        fig = go.Figure(
            go.Scatter(
                x=[r["analysed_at"] for r in individual],
                y=[r["overall_dis"] for r in individual],
                mode="markers",
                marker={
                    "size": 11,
                    "color": [RISK[r["risk_category"]][0] for r in individual],
                },
                text=[f"{r['batch_id']} · job #{r['job_id']}" for r in individual],
                hovertemplate="%{text}<br>DIS %{y:.3f}<br>%{x}<extra></extra>",
            )
        )
        fig.update_yaxes(range=[0, 1.1], title="Overall DIS")
        fig.update_xaxes(title="Processing timestamp")
        chart(
            layout(fig, 280, "All processing runs (timestamped)", legend=False),
            key="runs",
        )

    _combined_analysis([b["batch_id"] for b in batches])
    _processing_history()

    st.markdown("### Compare two populations")
    ids = [b["batch_id"] for b in batches]
    options = [BASELINE] + ids
    c1, c2 = st.columns(2)
    a = c1.selectbox("Population A", options, index=0, key="cmp_a")
    b = c2.selectbox("Population B", options, index=len(options) - 1, key="cmp_b")
    if a == b:
        st.warning("Choose two different populations.")
        return
    if BASELINE in (a, b):
        other = b if a == BASELINE else a
        meta = next(x for x in batches if x["batch_id"] == other)
        if not meta.get("latest_job_id"):
            st.info(f"{other} has not been analysed yet.")
            return
        report = safe(lambda: api.get(f"/analysis/{meta['latest_job_id']}"))
        if not report:
            return
        st.markdown(
            f"**{other} vs 2020 baseline** — DIS {fmt(report['overall_dis'])} {risk_badge(report['risk_category'])}",
            unsafe_allow_html=True,
        )
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
        )
        return

    with st.spinner("Comparing batches…"):
        cmp = safe(lambda: api.get("/compare", {"batch_a": a, "batch_b": b}))
    if not cmp:
        return
    rows = []
    for f in cmp["direct_drift"]:
        fa = cmp["features_a"].get(f["feature"], {})
        fb = cmp["features_b"].get(f["feature"], {})
        rows.append(
            {
                "Feature": f["label"],
                f"D vs baseline ({a})": fa.get("d"),
                f"D vs baseline ({b})": fb.get("d"),
                "Δ D": (
                    (fb.get("d") - fa.get("d"))
                    if fa.get("d") is not None and fb.get("d") is not None
                    else None
                ),
                f"DIS ({a})": fa.get("dis"),
                f"DIS ({b})": fb.get("dis"),
                "Direct PSI A→B": f["psi"],
                "Direct KS A→B": f["ks_stat"],
                "Direct JSD A→B": f["jsd"],
                "A→B status": f["drift_status"],
            }
        )
    df = pd.DataFrame(rows)
    num = {c: ".3f" for c in df.columns if c not in ("Feature", "A→B status")}
    table(df, num | {"Δ D": "+.2f"}, {"A→B status": status_cell})
    st.caption(
        "Left columns: each batch's drift against the 2020 baseline (from its latest analysis). Right columns: direct drift "
        "with batch A as the reference population, showing which attributes changed between the two batches."
    )
    feats = {f["label"]: f["feature"] for f in cmp["direct_drift"]}
    pick = st.selectbox("Feature distribution", list(feats), key="cmp_feat")
    chart(
        distribution_figure(
            cmp["distributions"][feats[pick]], f"{pick}: {a} vs {b}", 330
        ),
        key="cmp_dist",
    )


def _combined_analysis(ids: list[str]) -> None:
    st.markdown("### Combined batch analysis")
    st.caption(
        "Pools the selected batches into one customer population and compares it with the 2020 baseline. "
        "Individual batch results are kept unchanged."
    )
    picked = st.multiselect("Batches to combine", ids, default=ids, key="combine_ids")
    if st.button("Run combined analysis", disabled=len(picked) < 2 or not can_write()):
        with st.spinner("Pooling batches and comparing with the baseline…"):
            out = safe(
                lambda: api.post(
                    "/analysis/run", json={"batch_ids": picked, "mode": "combined"}
                )
            )
        if out:
            st.session_state["combined_job"] = out["reports"][0]["job_id"]
    job = st.session_state.get("combined_job")
    if not job:
        return
    report = safe(lambda: api.get(f"/analysis/{job}"))
    if not report:
        return
    st.markdown(
        f"**{report['batch_name']}** · {report['customer_count']:,} customers · DIS "
        f"{fmt(report['overall_dis'])} {risk_badge(report['risk_category'])} · job #{report['job_id']}",
        unsafe_allow_html=True,
    )
    table(
        feature_table(report),
        {
            "PSI": ".4f",
            "KS Statistic": ".4f",
            "KS p-value": ".2e",
            "JSD": ".4f",
            "D (normalised)": ".2f",
            "DIS": ".3f",
        },
        {"Drift Status": status_cell},
    )


def _processing_history() -> None:
    with st.expander("Processing history (monitoring jobs and activity log)"):
        logs = safe(lambda: api.get("/audit-logs", {"limit": 200}), {}) or {}
        jobs = pd.DataFrame(logs.get("jobs", []))
        if jobs.empty:
            st.caption("No monitoring jobs yet.")
        else:
            jobs = jobs.rename(
                columns={
                    "job_id": "Job",
                    "batch_id": "Batch",
                    "status": "Status",
                    "window_start": "Window start",
                    "window_end": "Window end",
                    "user": "Run by",
                    "alert_type": "Alert",
                    "alert_level": "Alert level",
                    "alert_created_at": "Alert raised",
                    "max_dis": "Max DIS",
                    "features_summarised": "Features",
                }
            )
            table(jobs, {"Max DIS": ".3f"}, height=300)
        events = logs.get("events", [])
        if events:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time": e["timestamp"],
                            "Event": e["event"],
                            "User": e.get("user_email"),
                            "Details": ", ".join(
                                f"{k}={v}" for k, v in (e.get("details") or {}).items()
                            ),
                        }
                        for e in events
                    ]
                ),
                hide_index=True,
                width="stretch",
                height=260,
            )
