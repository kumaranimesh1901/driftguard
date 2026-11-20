"""B. Dataset Management: baseline + production batch uploads, validation, preview and selection."""

import calendar
import hashlib
import re
from datetime import date
from pathlib import PurePath

import api_client as api
import pandas as pd
import streamlit as st
from components import fmt, header, note, risk_badge, safe, synthetic_notice, table

from sections.shared import FEATURE_LABELS, can_write, is_admin

MONTHS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]


def render() -> None:
    header(
        "Dataset Management",
        "Upload the 2020 historical baseline and 2026 production customer batches, then choose batches to analyse",
    )
    baseline_tab, batches_tab, analyse_tab = st.tabs(
        [
            "📁 1 · Historical Baseline",
            "📦 2 · Production Batches",
            "⚡ 3 · Select & Run Analysis",
        ]
    )
    with baseline_tab:
        _baseline_tab()
    with batches_tab:
        _batches_tab()
    with analyse_tab:
        _analyse_tab()
    if can_write():
        with st.expander("✨ Generate Synthetic Demo Data & Scenarios (Optional)", expanded=False):
            _demo_data()


def _baseline_tab() -> None:
    base = safe(lambda: api.get("/datasets/baseline"))
    if base is None:
        return
    if base.get("has_baseline"):
        _baseline_card(base)
    else:
        st.info(
            "No baseline registered. Upload the 2020 historical customer dataset below."
        )

    st.markdown("### Register a historical baseline")
    if not is_admin():
        st.caption("Only administrators can register a baseline. You have read access.")
        return
    note(
        "Required columns: customer_id, age, monthly_income, employment_type, credit_score, credit_limit, "
        "outstanding_loan_amount, fd_balance, debt_to_income, loan_amount_requested, loan_tenure, "
        "number_of_existing_loans, interest_rate. Optional: <b>loan_default</b> (0/1) — needed to train the baseline "
        "credit model, which supplies feature importance and sensitivity for DIS and enables model-health checks."
    )
    up = st.file_uploader("Baseline CSV", type=["csv"], key="baseline_file")
    c1, c2 = st.columns([2, 1])
    name = c1.text_input("Dataset name", "2020_Historical_Baseline", max_chars=120)
    fill = c2.checkbox(
        "Fill missing attributes with synthetic values",
        help="Absent required columns are generated "
        "synthetically and flagged. Real columns and real labels are never altered.",
    )
    if up is not None:
        payload = up.getvalue()
        key = hashlib.md5(payload).hexdigest() + str(fill)
        if st.session_state.get("baseline_validation_key") != key:
            with st.spinner("Validating…"):
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
            if st.button(
                "Register baseline", type="primary", disabled=not res["is_valid"]
            ):
                with st.spinner(
                    "Registering baseline and training the credit-risk model…"
                ):
                    out = safe(
                        lambda: api.post(
                            "/datasets/baseline",
                            files={"file": (up.name, payload, "text/csv")},
                            data={
                                "dataset_name": name,
                                "fill_missing_with_synthetic": str(fill).lower(),
                            },
                        )
                    )
                if out:
                    st.success(
                        f"Registered '{out['dataset_name']}' ({out['row_count']:,} customers). "
                        f"Model: {out['model']['algorithm']}."
                    )
                    st.rerun()

    with st.expander("Generate a synthetic 2020 baseline (demonstration only)"):
        st.caption(
            "Creates clearly flagged synthetic customers. Identical parameters reuse the existing baseline."
        )
        g1, g2 = st.columns(2)
        n = g1.number_input("Customers", 200, 50000, 2500, step=100)
        seed = g2.number_input("Random seed", 0, 2**31 - 1, 42)
        if st.button("Generate synthetic baseline"):
            out = safe(
                lambda: api.post(
                    "/simulation/generate-baseline",
                    json={"n": int(n), "seed": int(seed)},
                )
            )
            if out:
                st.success(
                    ("Reused" if out.get("reused") else "Registered")
                    + f" synthetic baseline '{out['dataset_name']}'."
                )
                st.rerun()


def _baseline_card(base: dict) -> None:
    synthetic_notice(base.get("is_synthetic"))
    m = base["model"]["metrics"]
    st.markdown(f"#### Active baseline: {base['dataset_name']}")
    c = st.columns(4)
    c[0].metric("Customers", f"{base['row_count']:,}")
    c[1].metric("Labelled outcomes", f"{base['labelled_rows']:,}")
    c[2].metric("Baseline default rate", fmt(m.get("baseline_default_rate"), ".1%"))
    c[3].metric("Model ROC-AUC (holdout)", fmt(m.get("roc_auc"), ".3f"))
    st.caption(
        f"Registered {base['created_at']} by {base['created_by']} · dataset #{base['dataset_id']} · "
        f"model #{base['model_id']} ({base['model']['algorithm']}, {base['model']['status']})"
    )
    if base.get("synthetic_columns"):
        st.caption("Synthetic columns: " + ", ".join(base["synthetic_columns"]))
    if not m.get("has_model"):
        st.warning(
            m.get("reason", "No model could be trained.")
            + " DIS will use normalised drift only."
        )

    t1, t2, t3 = st.tabs(["Reference profile", "Baseline model", "Record preview"])
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
                        "Std": r.get("std"),
                        "P5": q.get("0.05"),
                        "Median": q.get("0.5"),
                        "P95": q.get("0.95"),
                        "Missing": r.get("missing_rate"),
                    }
                )
            else:
                props = ", ".join(
                    f"{k} {v:.0%}" for k, v in (r.get("proportions") or {}).items()
                )
                rows.append(
                    {
                        "Feature": FEATURE_LABELS.get(f["name"], f["name"]),
                        "Type": f["type"],
                        "Mean": None,
                        "Std": None,
                        "P5": None,
                        "Median": None,
                        "P95": None,
                        "Missing": r.get("missing_rate"),
                        "Categories": props,
                    }
                )
        table(
            pd.DataFrame(rows).fillna({"Categories": ""}),
            {c: ",.2f" for c in ["Mean", "Std", "P5", "Median", "P95"]}
            | {"Missing": ".1%"},
            missing="",
        )
    with t2:
        if m.get("has_model"):
            mm = {
                k: m.get(k)
                for k in ("accuracy", "precision", "recall", "f1_score", "roc_auc")
            }
            st.dataframe(
                pd.DataFrame([mm]).rename(columns=str.title),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"{m.get('evaluation')}; trained on {m.get('n_train')} records, evaluated on {m.get('n_holdout')}; "
                f"decision threshold {m.get('threshold'):.2f} (F1-optimal on the training split)."
            )
    with t3:
        st.dataframe(
            pd.DataFrame(base.get("preview", [])), hide_index=True, width="stretch"
        )


def _validation_panel(res: dict) -> None:
    ok = res["is_valid"]
    (st.success if ok else st.error)(
        f"{res['filename']}: {res['row_count']:,} records · {res['column_count']} columns · "
        f"{res['labelled_rows']:,} labelled — {'valid' if ok else 'invalid'}"
    )
    for e in res["errors"]:
        st.error(e)
    if res["warnings"]:
        with st.expander(f"{len(res['warnings'])} warning(s)", expanded=not ok):
            for w in res["warnings"]:
                st.warning(w)
    st.dataframe(
        pd.DataFrame(res["preview"]), hide_index=True, width="stretch", height=220
    )


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
    return (
        "".join(c if c.isalnum() or c in "_-" else "_" for c in stem).strip("_-")
        or "batch"
    )[:64]


def _batches_tab() -> None:
    base = safe(lambda: api.get("/datasets/baseline"), {}) or {}
    if not base.get("has_baseline"):
        st.warning(
            "Register the historical baseline first. Batches can be uploaded now but analysed only after that."
        )
    if can_write():
        _upload_section()
    else:
        st.caption(
            "Reviewers have read-only access; uploads need the analyst or administrator role."
        )

    st.markdown("### Registered production batches")
    batches = safe(lambda: api.get("/batches"), []) or []
    if not batches:
        st.info("No production batches yet.")
        return
    synthetic_notice(any(b.get("is_synthetic") for b in batches))
    rows = []
    for b in batches:
        la = b.get("latest_analysis") or {}
        rows.append(
            {
                "Batch": b["batch_id"],
                "Batch date": b["batch_date"],
                "Customers": b["row_count"],
                "Labelled outcomes": b["labelled_rows"],
                "Source": "Synthetic" if b.get("is_synthetic") else "Uploaded",
                "Status": b["status"],
                "Overall DIS": la.get("overall_dis"),
                "Risk": la.get("risk_category"),
                "Uploaded": b["uploaded_at"],
                "Description": b.get("description"),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Overall DIS": st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.3f"
            )
        },
    )
    with st.expander("Preview batch records"):
        pick = st.selectbox(
            "Batch", [b["batch_id"] for b in batches], key="preview_batch"
        )
        detail = safe(lambda: api.get(f"/batches/{pick}"))
        if detail:
            st.caption(
                f"{detail['row_count']:,} records · file {detail.get('filename') or 'generated'} · "
                f"uploaded {detail['uploaded_at']} by {detail['uploaded_by']}"
            )
            st.dataframe(
                pd.DataFrame(detail["preview"]), hide_index=True, width="stretch"
            )


def _analyse_tab() -> None:
    batches = safe(lambda: api.get("/batches"), []) or []
    if not batches:
        st.info("Upload production batches first.")
        return
    ids = [b["batch_id"] for b in batches]
    pending = [b["batch_id"] for b in batches if b["status"] != "analysed"]
    st.markdown(
        "Each selected batch is compared individually against the 2020 baseline (PSI, KS, JSD, DIS, "
        "model health and risk recommendation). Results are saved and shown in the other sections."
    )
    selected = st.multiselect(
        "Production batches to analyse", ids, default=pending or ids, key="analyse_ids"
    )
    force = st.checkbox(
        "Re-process batches that were already analysed",
        help="By default, unchanged inputs return the stored result instead of recomputing.",
    )
    if not can_write():
        st.caption("Running analyses needs the analyst or administrator role.")
        return
    if st.button("Run drift analysis", type="primary", disabled=not selected):
        with st.spinner(f"Analysing {len(selected)} batch(es) in the backend…"):
            out = safe(
                lambda: api.post(
                    "/analysis/run",
                    json={"batch_ids": selected, "mode": "individual", "force": force},
                )
            )
        if out:
            for r in out["reports"]:
                st.markdown(
                    f"**{r['batch_id']}** — DIS {fmt(r['overall_dis'])} {risk_badge(r['risk_category'])}"
                    + (
                        " · stored result reused"
                        if r.get("reused")
                        else f" · new job #{r['job_id']}"
                    ),
                    unsafe_allow_html=True,
                )
            st.success(
                "Done. See Customer Drift Analysis, Drift Impact Score and Risk Recommendations."
            )
    st.caption(
        "To pool several batches into one population, use Batch Monitoring → Combined analysis."
    )


def _upload_section() -> None:
    st.markdown("### Upload batch files")
    files = st.file_uploader(
        "Production batch CSVs (select several, e.g. January_2026.csv … September_2026.csv)",
        type=["csv"],
        accept_multiple_files=True,
        key="batch_files",
    )
    if not files:
        return
    payloads = {f.name: f.getvalue() for f in files}
    key = hashlib.md5(
        b"".join(hashlib.md5(p).digest() for p in payloads.values())
    ).hexdigest()
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
    st.caption(
        "Review record counts and edit batch IDs, dates and descriptions before registering."
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
        label = f"{'✅' if res['is_valid'] else '❌'} {name} — {res['row_count']:,} records, {len(res['errors'])} error(s), {len(res['warnings'])} warning(s)"
        with st.expander(label, expanded=not res["is_valid"]):
            _validation_panel(res)

    c1, c2, c3 = st.columns(3)
    overwrite = c1.checkbox("Overwrite existing batch IDs")
    allow_dup = c2.checkbox("Allow duplicate file content")
    auto = c3.checkbox("Run individual analysis after upload", value=True)
    valid_count = int(edited["Valid"].sum())
    if st.button(
        f"Register {valid_count} valid batch(es)",
        type="primary",
        disabled=valid_count == 0,
    ):
        rows = edited[edited["Valid"]]
        files_payload = [
            ("files", (r["File"], payloads[r["File"]], "text/csv"))
            for _, r in rows.iterrows()
        ]
        form = {
            "batch_ids": [str(r["Batch ID"]) for _, r in rows.iterrows()],
            "batch_dates": [
                str(pd.Timestamp(r["Batch date"]).date()) for _, r in rows.iterrows()
            ],
            "descriptions": [str(r["Description"] or "") for _, r in rows.iterrows()],
            "overwrite": str(overwrite).lower(),
            "allow_duplicate_content": str(allow_dup).lower(),
        }
        with st.spinner("Uploading…"):
            out = safe(
                lambda: api.post("/batches/upload", files=files_payload, data=form)
            )
        if out:
            for r in out["results"]:
                if r["status"] == "registered":
                    st.success(
                        f"{r['batch_id']}: registered {r['row_count']:,} records ({r['batch_date']})."
                    )
                else:
                    st.error(f"{r['batch_id']}: rejected — {r['error']}")
            registered = [
                r["batch_id"] for r in out["results"] if r["status"] == "registered"
            ]
            if auto and registered:
                with st.spinner(f"Analysing {len(registered)} batch(es)…"):
                    ana = safe(
                        lambda: api.post(
                            "/analysis/run",
                            json={
                                "batch_ids": registered,
                                "mode": "individual",
                                "force": overwrite,
                            },
                        )
                    )
                if ana:
                    for r in ana["reports"]:
                        st.markdown(
                            f"**{r['batch_id']}** analysed — DIS {fmt(r['overall_dis'])} {risk_badge(r['risk_category'])}",
                            unsafe_allow_html=True,
                        )


def _demo_data() -> None:
    if is_admin():
        st.markdown("##### Seed the complete demo")
        st.caption(
            "Registers a synthetic 2020 baseline (if none is identical) and Jan/Mar/Jun/Sep 2026 batches "
            "(scenarios A–D), plus an optional labelled Nov 2026 concept-shift batch, then analyses them."
        )
        concept = st.checkbox("Include Nov 2026 concept-shift batch", value=True)
        if st.button("Seed synthetic demo"):
            with st.spinner("Generating data and running analyses…"):
                out = safe(
                    lambda: api.post(
                        "/simulation/seed-demo",
                        json={"include_concept_shift_batch": concept},
                    )
                )
            if out:
                st.success(f"Seeded and analysed {len(out['analyses'])} batches.")
    st.markdown("##### Generate one scenario batch")
    st.caption(
        "For demonstrations only. Generated batches are permanently flagged as synthetic and never presented as real customers."
    )
    scenarios = safe(lambda: api.get("/simulation/scenarios"), {}) or {}
    if not scenarios:
        return
    with st.form("gen"):
        c1, c2, c3 = st.columns(3)
        sc = c1.selectbox(
            "Scenario",
            list(scenarios),
            format_func=lambda s: f"{s} — {scenarios[s]['name']}",
        )
        bid = c2.text_input("Batch ID", "Synthetic_Q4_2026", max_chars=64)
        bdate = c3.date_input("Batch date", date(2026, 12, 31))
        c4, c5, c6, c7 = st.columns(4)
        n = c4.number_input("Customers", 100, 20000, 1200, step=100)
        labels = c5.checkbox("Include default outcome labels")
        concept = c6.checkbox(
            "Add concept shift",
            help="Changes the true default relationship in 2026 so labelled "
            "outcomes can reveal genuine performance degradation.",
        )
        overwrite = c7.checkbox("Overwrite if exists")
        st.caption(
            f"Expected: {scenarios[sc]['expected']}. {scenarios[sc]['description']}"
        )
        if st.form_submit_button("Generate and register synthetic batch"):
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
                    "Run analysis above."
                )
