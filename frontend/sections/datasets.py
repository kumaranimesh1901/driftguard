"""B. Dataset Management: baseline card, feature schema, CSV uploads, and smart synthetic generator."""

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
    empty_state,
    fmt,
    format_date,
    format_timestamp,
    info_callout,
    kpi_card,
    page_header,
    risk_badge,
    safe,
    section,
    status_pill,
    synthetic_notice,
    table,
)

from sections.shared import FEATURE_LABELS, can_write, is_admin, visible_batches

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def render() -> None:
    page_header(
        "Dataset Management",
        "Manage the 2020 historical baseline, feature schema, production batch uploads and synthetic demos",
        show_batch_selector=True,
    )
    baseline_tab, batches_tab, analyse_tab = st.tabs(
        ["1 · Historical Baseline", "2 · Production Batches", "3 · Select & Run Analysis"]
    )
    with baseline_tab:
        _baseline_tab()
    with batches_tab:
        _batches_tab()
    with analyse_tab:
        _analyse_tab()
    if can_write():
        with st.expander("Generate Synthetic Demo Data & Scenarios (Optional)", expanded=False):
            _demo_data()


def _baseline_tab() -> None:
    base = safe(lambda: api.get("/datasets/baseline"), retry_key="ds_base")
    if base is None:
        return
    if base.get("has_baseline"):
        _baseline_card(base)
    else:
        empty_state(
            "NO BASELINE",
            "No historical baseline registered",
            "Upload the 2020 historical customer CSV below to establish the reference distribution and champion model.",
        )

    section("Register a Historical Baseline")
    if not is_admin():
        info_callout("info", "Only administrators can register a baseline. You currently have read access.")
        return
    info_callout(
        "info",
        "Required columns: <code>customer_id, age, monthly_income, employment_type, credit_score, credit_limit, "
        "outstanding_loan_amount, fd_balance, debt_to_income, loan_amount_requested, loan_tenure, "
        "number_of_existing_loans, interest_rate</code>. Optional: <b>loan_default</b> (0/1) to train the credit model.",
    )
    up = st.file_uploader("Baseline CSV", type=["csv"], key="baseline_file")
    c1, c2 = st.columns([2, 1])
    name = c1.text_input("Dataset name", "2020_Historical_Baseline", max_chars=120)
    fill = c2.checkbox(
        "Fill missing attributes with synthetic values",
        help="Absent required columns are generated synthetically and flagged. Real columns and labels are never altered.",
    )
    if up is not None:
        payload = up.getvalue()
        key = hashlib.md5(payload).hexdigest() + str(fill)
        if st.session_state.get("baseline_validation_key") != key:
            with st.spinner("Validating baseline CSV…"):
                res = safe(
                    lambda: api.post(
                        "/datasets/validate",
                        files={"file": (up.name, payload, "text/csv")},
                        data={"fill_missing_with_synthetic": str(fill).lower()},
                    )
                )
            st.session_state["baseline_validation"] = res
            st.session_state["baseline_validation_key"] = key
        res = st.session_state.get("baseline_validation")
        if res:
            _validation_panel(res)
            if st.button("Register baseline", type="primary", disabled=not res["is_valid"]):
                with st.spinner("Registering baseline and training the credit-risk model…"):
                    out = safe(
                        lambda: api.post(
                            "/datasets/baseline",
                            files={"file": (up.name, payload, "text/csv")},
                            data={"dataset_name": name, "fill_missing_with_synthetic": str(fill).lower()},
                        )
                    )
                if out:
                    st.success(f"Registered '{out['dataset_name']}' ({out['row_count']:,} customers).")
                    st.rerun()

    with st.expander("Generate a synthetic 2020 baseline (demonstration only)"):
        st.caption("Creates clearly flagged synthetic customers. Identical parameters reuse the existing baseline.")
        g1, g2 = st.columns(2)
        n = g1.number_input("Customers", 200, 50000, 2500, step=100)
        seed = g2.number_input("Random seed", 0, 2**31 - 1, 42)
        if st.button("Generate synthetic baseline"):
            out = safe(lambda: api.post("/simulation/generate-baseline", json={"n": int(n), "seed": int(seed)}))
            if out:
                st.success(
                    ("Reused" if out.get("reused") else "Registered")
                    + f" synthetic baseline '{out['dataset_name']}'."
                )
                st.rerun()


def _baseline_card(base: dict) -> None:
    m = base["model"]["metrics"]
    synth_pill = status_pill("Synthetic data", "neutral", dot=False) if base.get("is_synthetic") else ""
    st.markdown(
        f"""
        <div class="dg-card" style="border-left:4px solid {theme.PRIMARY};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div style="font-size:16px;font-weight:700;color:{theme.INK};">Active Baseline: {base['dataset_name']} &nbsp;{synth_pill}</div>
                <div style="font-size:12px;color:{theme.MUTED};">Registered {format_timestamp(base.get('created_at'))} by {base['created_by']} · model #{base['model_id']} ({base['model']['status']})</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c = st.columns(5)
    kpi_card(c[0], "Baseline Rows", f"{base['row_count']:,}", "historical customers", tone="primary")
    kpi_card(c[1], "Monitored Features", str(len(base["features"])), "banking attributes", tone="primary")
    kpi_card(c[2], "Labelled Outcomes", f"{base['labelled_rows']:,}", "loan_default labels", tone="primary")
    kpi_card(c[3], "Default Rate", fmt(m.get("baseline_default_rate"), ".1%"), "baseline actual", tone="neutral")
    kpi_card(c[4], "Holdout ROC-AUC", fmt(m.get("roc_auc"), ".3f"), base["model"]["algorithm"], tone="primary")

    if not m.get("has_model"):
        info_callout("warning", m.get("reason", "No model could be trained.") + " DIS will use normalised drift only.")

    t1, t2, t3 = st.tabs(["Feature Schema & Reference Stats", "Baseline Model", "Record Preview"])
    with t1:
        rows = []
        for f in base["features"]:
            r = f["reference_stats"] or {}
            if f["type"] == "numerical":
                q = r.get("quantiles", {})
                rows.append(
                    {
                        "Feature": FEATURE_LABELS.get(f["name"], f["name"]),
                        "Type": f["type"],
                        "Mean": r.get("mean"),
                        "Median": q.get("0.5"),
                        "Std": r.get("std"),
                        "Missing %": r.get("missing_rate"),
                    }
                )
            else:
                props = ", ".join(f"{k} {v:.0%}" for k, v in (r.get("proportions") or {}).items())
                rows.append(
                    {
                        "Feature": FEATURE_LABELS.get(f["name"], f["name"]),
                        "Type": f["type"],
                        "Mean": None,
                        "Median": None,
                        "Std": None,
                        "Missing %": r.get("missing_rate"),
                        "Categories": props,
                    }
                )
        table(
            pd.DataFrame(rows).fillna({"Categories": ""}),
            {"Mean": ",.2f", "Median": ",.2f", "Std": ",.2f", "Missing %": ".1%"},
            missing="",
        )
    with t2:
        if m.get("has_model"):
            mm = {k: m.get(k) for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")}
            st.dataframe(pd.DataFrame([mm]).rename(columns=str.title), hide_index=True, width="stretch")
            st.caption(
                f"{m.get('evaluation')} · trained on {m.get('n_train'):,}, evaluated on {m.get('n_holdout'):,} · "
                f"threshold {m.get('threshold'):.2f}"
            )
    with t3:
        st.dataframe(pd.DataFrame(base.get("preview", [])), hide_index=True, width="stretch")


def _validation_panel(res: dict) -> None:
    ok = res["is_valid"]
    info_callout(
        "success" if ok else "danger",
        f"<b>{res['filename']}</b>: {res['row_count']:,} records · {res['column_count']} columns · "
        f"{res['labelled_rows']:,} labelled — <b>{'Valid' if ok else 'Invalid'}</b>",
    )
    for e in res["errors"]:
        st.error(e)
    if res["warnings"]:
        with st.expander(f"{len(res['warnings'])} warning(s)", expanded=not ok):
            for w in res["warnings"]:
                st.warning(w)
    st.dataframe(pd.DataFrame(res["preview"]), hide_index=True, width="stretch", height=220)


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


def _batches_tab() -> None:
    base = safe(lambda: api.get("/datasets/baseline"), {}) or {}
    if not base.get("has_baseline"):
        info_callout("warning", "Register the historical baseline first before running analyses.")
    if can_write():
        _upload_section()
    else:
        info_callout("info", "Reviewers have read-only access; uploads require the analyst or administrator role.")

    section("Registered Production Batches")
    batches = visible_batches(safe(lambda: api.get("/batches"), []) or [])
    if not batches:
        empty_state("NO BATCHES", "No production batches yet", "Upload batch CSVs above or generate synthetic scenarios.")
        return
    synthetic_notice(any(b.get("is_synthetic") for b in batches))
    rows = []
    for b in batches:
        la = b.get("latest_analysis") or {}
        rows.append(
            {
                "Batch": b["batch_id"],
                "Date": format_date(b["batch_date"]),
                "Customers": b["row_count"],
                "Labelled": b["labelled_rows"],
                "Source": "Synthetic" if b.get("is_synthetic") else "Uploaded",
                "Status": b["status"].title(),
                "Overall DIS": la.get("overall_dis"),
                "Risk": la.get("risk_category") or "Not analysed",
                "Uploaded": format_timestamp(b["uploaded_at"]),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Overall DIS": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")
        },
    )
    with st.expander("Preview batch records"):
        pick = st.selectbox("Batch", [b["batch_id"] for b in batches], key="preview_batch")
        detail = safe(lambda: api.get(f"/batches/{pick}"))
        if detail:
            st.caption(
                f"{detail['row_count']:,} records · file {detail.get('filename') or 'generated'} · "
                f"uploaded {format_timestamp(detail['uploaded_at'])} by {detail['uploaded_by']}"
            )
            st.dataframe(pd.DataFrame(detail["preview"]), hide_index=True, width="stretch")


def _analyse_tab() -> None:
    batches = visible_batches(safe(lambda: api.get("/batches"), []) or [])
    if not batches:
        empty_state("NO BATCHES", "Upload production batches first", "Once registered, batches appear here for analysis.")
        return
    ids = [b["batch_id"] for b in batches]
    pending = [b["batch_id"] for b in batches if b["status"] != "analysed"]
    st.caption(
        "Each selected batch is compared individually against the 2020 baseline (PSI, KS, JSD, DIS, ADWIN, model health & risk)."
    )
    selected = st.multiselect("Production batches to analyse", ids, default=pending or ids, key="analyse_ids")
    force = st.checkbox(
        "Re-process batches that were already analysed",
        help="By default, unchanged inputs return the stored result instead of recomputing.",
    )
    if not can_write():
        info_callout("info", "Running analyses requires the analyst or administrator role.")
        return
    if st.button("Run drift analysis", type="primary", disabled=not selected):
        with st.spinner(f"Analysing {len(selected)} batch(es) in the backend…"):
            out = safe(
                lambda: api.post("/analysis/run", json={"batch_ids": selected, "mode": "individual", "force": force})
            )
        if out:
            for r in out["reports"]:
                st.markdown(
                    f"**{r['batch_id']}** — DIS {fmt(r['overall_dis'])} {risk_badge(r['risk_category'])}"
                    + (" · cached result reused" if r.get("reused") else f" · job #{r['job_id']}"),
                    unsafe_allow_html=True,
                )
            st.success("Analysis complete.")


def _upload_section() -> None:
    section("Upload Batch Files")
    files = st.file_uploader(
        "Production batch CSVs (select one or more)",
        type=["csv"],
        accept_multiple_files=True,
        key="batch_files",
    )
    if not files:
        return
    payloads = {f.name: f.getvalue() for f in files}
    key = hashlib.md5(b"".join(hashlib.md5(p).digest() for p in payloads.values())).hexdigest()
    if st.session_state.get("batch_validation_key") != key:
        results = {}
        with st.spinner(f"Validating {len(files)} file(s)…"):
            for name, payload in payloads.items():
                results[name] = safe(
                    lambda n=name, p=payload: api.post(
                        "/datasets/validate", files={"file": (n, p, "text/csv")}
                    )
                )
        st.session_state["batch_validation"] = results
        st.session_state["batch_validation_key"] = key
    results = st.session_state["batch_validation"]

    meta = pd.DataFrame(
        [
            {
                "File": name,
                "Records": (results.get(name) or {}).get("row_count"),
                "Labelled": (results.get(name) or {}).get("labelled_rows"),
                "Valid": bool((results.get(name) or {}).get("is_valid")),
                "Batch ID": _batch_id(name),
                "Batch date": _infer_date(name),
                "Description": "",
            }
            for name in payloads
        ]
    )
    edited = st.data_editor(
        meta,
        hide_index=True,
        width="stretch",
        key=f"meta_{key}",
        disabled=["File", "Records", "Labelled", "Valid"],
        column_config={
            "Batch date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Description": st.column_config.TextColumn(max_chars=300),
        },
    )
    for name in payloads:
        res = results.get(name)
        if not res:
            continue
        lbl = f"{'Valid' if res['is_valid'] else 'Invalid'} · {name} ({res['row_count']:,} records, {len(res['errors'])} errors)"
        with st.expander(lbl, expanded=not res["is_valid"]):
            _validation_panel(res)

    c1, c2, c3 = st.columns(3)
    overwrite = c1.checkbox("Overwrite existing batch IDs")
    allow_dup = c2.checkbox("Allow duplicate file content")
    auto = c3.checkbox("Run individual analysis after upload", value=True)
    valid_count = int(edited["Valid"].sum())
    if st.button(f"Register {valid_count} valid batch(es)", type="primary", disabled=valid_count == 0):
        rows = edited[edited["Valid"]]
        files_payload = [("files", (r["File"], payloads[r["File"]], "text/csv")) for _, r in rows.iterrows()]
        form = {
            "batch_ids": [str(r["Batch ID"]) for _, r in rows.iterrows()],
            "batch_dates": [str(pd.Timestamp(r["Batch date"]).date()) for _, r in rows.iterrows()],
            "descriptions": [str(r["Description"] or "") for _, r in rows.iterrows()],
            "overwrite": str(overwrite).lower(),
            "allow_duplicate_content": str(allow_dup).lower(),
        }
        with st.spinner("Uploading…"):
            out = safe(lambda: api.post("/batches/upload", files=files_payload, data=form))
        if out:
            for r in out["results"]:
                if r["status"] == "registered":
                    st.success(f"{r['batch_id']}: registered {r['row_count']:,} records ({r['batch_date']}).")
                else:
                    st.error(f"{r['batch_id']}: rejected — {r['error']}")
            registered = [r["batch_id"] for r in out["results"] if r["status"] == "registered"]
            if auto and registered:
                with st.spinner(f"Analysing {len(registered)} batch(es)…"):
                    safe(
                        lambda: api.post(
                            "/analysis/run",
                            json={"batch_ids": registered, "mode": "individual", "force": overwrite},
                        )
                    )


def _suggested_synth_id(scenario: str, bdate: date) -> str:
    return f"Synthetic_{scenario}_{bdate.strftime('%Y-%m')}"


def _mentioned_scenario_letter(batch_id: str) -> str | None:
    """Detect an explicit scenario letter (A/B/C/D) embedded in a batch ID."""
    m = re.search(r"(?:scenario[_\-\s]*|synthetic[_\-])([A-D])(?![a-zA-Z])", batch_id, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _demo_data() -> None:
    if is_admin():
        st.markdown("##### Seed the complete demo")
        st.caption(
            "Registers a synthetic 2020 baseline (if none is identical) and Jan/Mar/Jun/Sep/Nov 2026 batches, then analyses them."
        )
        concept = st.checkbox("Include Nov 2026 concept-shift batch", value=True)
        if st.button("Seed synthetic demo"):
            with st.spinner("Generating data and running analyses…"):
                out = safe(lambda: api.post("/simulation/seed-demo", json={"include_concept_shift_batch": concept}))
            if out:
                st.success(f"Seeded and analysed {len(out['analyses'])} batches.")

    st.markdown("##### Generate one scenario batch")
    st.caption(
        "For demonstrations only. Generated batches are permanently flagged as synthetic."
    )
    scenarios = safe(lambda: api.get("/simulation/scenarios"), {}) or {}
    if not scenarios:
        return
    sc_list = list(scenarios)

    c1, c2, c3 = st.columns(3)
    sc = c1.selectbox(
        "Scenario",
        sc_list,
        format_func=lambda s: f"{s} — {scenarios[s]['name']}",
        key="gen_sc",
    )
    bdate = c3.date_input("Batch date", value=date.today(), key="gen_bdate")

    # Auto-prefill Batch ID from scenario + date unless the user manually edited it
    suggested = _suggested_synth_id(sc, bdate)
    last_suggested = st.session_state.get("_gen_last_suggested")
    if "gen_bid" not in st.session_state or st.session_state.get("gen_bid") == last_suggested:
        st.session_state["gen_bid"] = suggested
    st.session_state["_gen_last_suggested"] = suggested

    bid = c2.text_input("Batch ID", max_chars=64, key="gen_bid")

    # Non-blocking warnings per spec
    mentioned = _mentioned_scenario_letter(bid)
    if mentioned and mentioned != sc.upper():
        info_callout(
            "warning",
            f"<b>Batch ID / Scenario mismatch:</b> Batch ID <code>{bid}</code> mentions scenario <b>{mentioned}</b>, "
            f"but scenario <b>{sc}</b> ({scenarios[sc]['name']}) is selected.",
        )
    if bdate > date.today():
        info_callout(
            "warning",
            f"<b>Future batch date ({format_date(bdate.isoformat())}):</b> This date is later than today, "
            "so this batch will become the <b>latest batch</b> shown in the Overview hero banner.",
        )

    c4, c5, c6, c7 = st.columns(4)
    n = c4.number_input("Customers", 100, 20000, 1200, step=100, key="gen_n")
    labels = c5.checkbox("Include default outcome labels", key="gen_labels")
    concept = c6.checkbox(
        "Add concept shift",
        help="Changes the true default relationship so labelled outcomes reveal genuine performance degradation.",
        key="gen_concept",
    )
    overwrite = c7.checkbox("Overwrite if exists", key="gen_overwrite")
    st.caption(f"Expected: {scenarios[sc]['expected']}. {scenarios[sc]['description']}")
    if st.button("Generate and register synthetic batch", type="primary", key="gen_submit"):
        out = safe(
            lambda: api.post(
                "/simulation/generate-batch",
                json={
                    "scenario": sc,
                    "batch_id": bid,
                    "batch_date": bdate.isoformat(),
                    "n": int(n),
                    "with_labels": labels,
                    "concept_shift": concept,
                    "overwrite": overwrite,
                },
            )
        )
        if out:
            st.success(
                f"Registered synthetic batch {out['batch_id']} ({out['row_count']:,} records, scenario {out['scenario']}). "
                "Run analysis in tab 3."
            )
