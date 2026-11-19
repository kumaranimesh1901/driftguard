"""H. Reports: downloadable PDF drift monitoring report."""

import api_client as api
import pandas as pd
import streamlit as st
from components import (
    fmt,
    header,
    safe,
)

from sections.shared import select_analysis


def render() -> None:
    header(
        "Reports",
        "Generate a PDF pack with drift metrics, DIS, model performance and recommended actions",
    )
    choice = select_analysis("report_job")
    if not choice:
        return
    c = st.columns(4)
    c[0].metric("Customers", f"{choice['customer_count']:,}")
    c[1].metric("Overall DIS", fmt(choice["overall_dis"]))
    c[2].metric("Risk", choice["risk_category"])
    c[3].metric("Performance", choice["performance_state"])
    st.caption(
        f"Job #{choice['job_id']} · analysed {choice['analysed_at']} by {choice['analysed_by']}"
    )
    if st.button("Generate PDF report", type="primary"):
        with st.spinner("Rendering PDF…"):
            pdf = safe(lambda: api.get_bytes(f"/reports/{choice['job_id']}/pdf"))
        if pdf:
            st.session_state["pdf"] = (choice["job_id"], pdf)
    stored = st.session_state.get("pdf")
    if stored and stored[0] == choice["job_id"]:
        safe_name = "".join(
            ch if ch.isalnum() or ch in "_-" else "_" for ch in choice["batch_id"]
        )
        st.download_button(
            "Download PDF",
            stored[1],
            file_name=f"DriftGuard_Risk_Report_{safe_name}_job{choice['job_id']}.pdf",
            mime="application/pdf",
        )
        st.success(f"Report ready ({len(stored[1]) / 1024:.0f} KB).")
    st.markdown("#### Report contents")
    st.markdown(
        "1. Monitoring summary · 2. Historical baseline · 3. Production batch information · 4. Customer record counts · "
        "5. Feature-level PSI, KS and JSD · 6. Drift Impact Score with chart · 7. Highest-risk features with distribution chart · "
        "8. Model performance (when labels exist) · 9. Risk assessment · 10. Recommended actions · 11. Monitoring timestamp"
    )
    logs = (
        safe(
            lambda: api.get("/audit-logs", {"event": "report_generated", "limit": 20}),
            {},
        )
        or {}
    )
    if logs.get("events"):
        st.markdown("#### Recently generated reports")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Generated": e["timestamp"],
                        "By": e["user_email"],
                        "Job": e["details"].get("job_id"),
                        "Batch": e["details"].get("batch_id"),
                    }
                    for e in logs["events"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
