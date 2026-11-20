"""E. Batch Monitoring: sortable batch table, drag-and-drop upload card, comparisons & history."""

import calendar
import hashlib
import re
from datetime import date
from pathlib import PurePath

import api_client as api
import pandas as pd
import streamlit as st
import theme
from components import (
    chart,
    distribution_figure,
    empty_state,
    fmt,
    format_date,
    format_perf_state,
    format_timestamp,
    info_callout,
    page_header,
    risk_badge,
    safe,
    section,
    status_cell,
    table,
)

from sections.shared import can_write, feature_table, visible_batches

BASELINE = "Baseline 2020"
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _infer_date(filename: str) -> date:
    stem = PurePath(filename).stem.lower()
    year = re.search(r"(20\d{2})", stem)
    month = next((i + 1 for i, m in enumerate(MONTHS) if m in stem), None)
    if year and month:
        y = int(year.group(1))
        return date(y, month, calendar.monthrange(y, month)[1])
    return date.today()


def _batch_id(filename: str) -> str:
    stem = PurePath(filename).stem
    return ("".join(c if c.isalnum() or c in "_-" else "_" for c in stem).strip("_-") or "batch")[:64]


def render() -> None:
    page_header(
        "Batch Monitoring",
        "Sortable ledger of production batches, quick CSV ingestion, population comparison and audit history",
        show_batch_selector=True,
    )
    all_batches = safe(lambda: api.get("/batches"), [], retry_key="bm_batches") or []
    batches = visible_batches(all_batches)

    # 1. Sortable Batch Table
    section(
        f"Production Batches ({len(batches)})",
        "Click any column header to sort; DIS progress bar scales 0 → 1.",
    )
    if not batches:
        empty_state(
            "NO BATCHES",
            "No production batches registered",
            "Upload a production CSV below or generate synthetic scenarios in Dataset Management.",
        )
    else:
        rows = []
        for b in batches:
            la = b.get("latest_analysis") or {}
            rows.append(
                {
                    "Batch": b["batch_id"],
                    "Date": format_date(b.get("batch_date")),
                    "Rows": b.get("row_count", 0),
                    "Labelled": b.get("labelled_rows", 0),
                    "Status": (b.get("status") or "pending").title(),
                    "DIS": la.get("overall_dis"),
                    "Risk": la.get("risk_category") or "Not analysed",
                    "Performance": format_perf_state(la.get("performance_state")),
                    "Source": "Synthetic" if b.get("is_synthetic") else "Uploaded",
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Rows": st.column_config.NumberColumn("Rows", format="%d"),
                "Labelled": st.column_config.NumberColumn("Labelled", format="%d"),
                "DIS": st.column_config.ProgressColumn(
                    "DIS", min_value=0.0, max_value=1.0, format="%.3f", help=theme.HELP_TEXTS["dis"]
                ),
            },
        )

    # 2. Upload Card (drag-and-drop)
    section("Upload Production Batch", "Drag and drop a CSV to validate schema and register a new batch")
    if not can_write():
        info_callout("info", "Reviewers have read-only access. Uploading batches requires the analyst or administrator role.")
    else:
        _quick_upload_card()

    if not batches:
        return

    # 3. Secondary Tools in Clean Tabs
    tab_cmp, tab_comb, tab_hist = st.tabs(
        ["Compare Two Populations", "Combined Batch Analysis", "Processing History"]
    )
    with tab_cmp:
        _compare_populations(batches)
    with tab_comb:
        _combined_analysis([b["batch_id"] for b in batches])
    with tab_hist:
        _processing_history()


def _quick_upload_card() -> None:
    up = st.file_uploader(
        "Drop a production batch CSV here",
        type=["csv"],
        key="bm_upload_csv",
        help="Required banking attributes are validated automatically before registration.",
    )
    if up is None:
        return
    payload = up.getvalue()
    key = hashlib.md5(payload).hexdigest()
    if st.session_state.get("bm_val_key") != key:
        with st.spinner("Validating CSV schema…"):
            res = safe(
                lambda: api.post(
                    "/datasets/validate", files={"file": (up.name, payload, "text/csv")}
                )
            )
        st.session_state["bm_val"] = res
        st.session_state["bm_val_key"] = key
    res = st.session_state.get("bm_val")
    if not res:
        return

    ok = res["is_valid"]
    info_callout(
        "success" if ok else "danger",
        f"<b>{res['filename']}</b> — {res['row_count']:,} records · {res['column_count']} columns · "
        f"{res['labelled_rows']:,} labelled · <b>{'Valid' if ok else 'Invalid'}</b>",
    )
    for err in res.get("errors", []):
        st.error(err)
    for warn in res.get("warnings", []):
        st.warning(warn)

    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    bid = c1.text_input("Batch ID", value=_batch_id(up.name), key=f"bm_bid_{key}")
    bdate = c2.date_input("Batch date", value=_infer_date(up.name), key=f"bm_bdate_{key}")
    auto = c3.checkbox("Analyse after upload", value=True, key=f"bm_auto_{key}")
    if st.button("Register batch", type="primary", disabled=not ok, key=f"bm_reg_{key}"):
        form = {
            "batch_ids": [bid],
            "batch_dates": [bdate.isoformat()],
            "descriptions": [""],
            "overwrite": "false",
            "allow_duplicate_content": "false",
        }
        with st.spinner("Uploading batch…"):
            out = safe(
                lambda: api.post(
                    "/batches/upload",
                    files=[("files", (up.name, payload, "text/csv"))],
                    data=form,
                )
            )
        if out and out.get("results"):
            r0 = out["results"][0]
            if r0["status"] == "registered":
                st.success(f"Registered {r0['batch_id']} ({r0['row_count']:,} rows).")
                if auto:
                    with st.spinner("Running drift analysis…"):
                        safe(lambda: api.post("/analysis/run", json={"batch_ids": [r0["batch_id"]]}))
                st.rerun()
            else:
                st.error(f"Rejected: {r0.get('error')}")


def _compare_populations(batches: list[dict]) -> None:
    ids = [b["batch_id"] for b in batches]
    options = [BASELINE] + ids
    c1, c2 = st.columns(2)
    a = c1.selectbox("Population A", options, index=0, key="cmp_a")
    b = c2.selectbox("Population B", options, index=len(options) - 1, key="cmp_b")
    if a == b:
        info_callout("warning", "Choose two different populations to compare.")
        return
    if BASELINE in (a, b):
        other = b if a == BASELINE else a
        meta = next(x for x in batches if x["batch_id"] == other)
        if not meta.get("latest_job_id"):
            empty_state("PENDING", f"{other} has not been analysed yet", "Run an analysis on this batch first.")
            return
        report = safe(lambda: api.get(f"/analysis/{meta['latest_job_id']}"))
        if not report:
            return
        st.markdown(
            f"**{other} vs 2020 baseline** — DIS {fmt(report['overall_dis'])} {risk_badge(report['risk_category'])}",
            unsafe_allow_html=True,
        )
        table(
            feature_table(report),
            {"PSI": ".4f", "KS Statistic": ".4f", "KS p-value": ".2e", "JSD": ".4f", "D (normalised)": ".3f", "DIS": ".3f"},
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
                f"D ({a})": fa.get("d"),
                f"D ({b})": fb.get("d"),
                "Δ D": ((fb.get("d") - fa.get("d")) if fa.get("d") is not None and fb.get("d") is not None else None),
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
    table(df, num | {"Δ D": "+.3f"}, {"A→B status": status_cell})
    feats = {f["label"]: f["feature"] for f in cmp["direct_drift"]}
    pick = st.selectbox("Feature distribution", list(feats), key="cmp_feat")
    chart(distribution_figure(cmp["distributions"][feats[pick]], f"{pick}: {a} vs {b}", 320), key="cmp_dist")


def _combined_analysis(ids: list[str]) -> None:
    st.caption("Pool several batches into one population and compare against the 2020 baseline.")
    picked = st.multiselect("Batches to combine", ids, default=ids, key="combine_ids")
    if st.button("Run combined analysis", disabled=len(picked) < 2 or not can_write()):
        with st.spinner("Pooling batches and analysing…"):
            out = safe(lambda: api.post("/analysis/run", json={"batch_ids": picked, "mode": "combined"}))
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
        {"PSI": ".4f", "KS Statistic": ".4f", "KS p-value": ".2e", "JSD": ".4f", "D (normalised)": ".3f", "DIS": ".3f"},
        {"Drift Status": status_cell},
    )


def _processing_history() -> None:
    logs = safe(lambda: api.get("/audit-logs", {"limit": 100}), {}) or {}
    jobs = pd.DataFrame(logs.get("jobs", []))
    if jobs.empty:
        empty_state("NO JOBS", "No monitoring jobs recorded yet", "Run an analysis to populate job history.")
        return
    jobs = jobs.rename(
        columns={
            "job_id": "Job",
            "batch_id": "Batch",
            "status": "Status",
            "user": "Run by",
            "alert_type": "Alert",
            "alert_level": "Level",
            "max_dis": "Max DIS",
        }
    )
    cols = [c for c in ("Job", "Batch", "Status", "Max DIS", "Alert", "Level", "Run by") if c in jobs.columns]
    table(jobs[cols], {"Max DIS": ".3f"}, height=300)
    events = logs.get("events", [])
    if events:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Time": format_timestamp(e.get("timestamp")),
                        "Event": e["event"],
                        "User": e.get("user_email"),
                        "Details": ", ".join(f"{k}={v}" for k, v in (e.get("details") or {}).items()),
                    }
                    for e in events[:40]
                ]
            ),
            hide_index=True,
            width="stretch",
            height=240,
        )
