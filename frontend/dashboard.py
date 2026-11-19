"""DriftGuard — Banking Customer Drift Monitoring Dashboard.

The only user-facing application. Every statistic (PSI, KS, JSD, DIS, model
health, recommendations, reports) is computed by the FastAPI backend; this
file only handles sign-in and navigation between the eight sections.

Run via ./start.sh (or: streamlit run frontend/dashboard.py).
"""

import api_client as api
import streamlit as st
from components import apply_theme, safe
from sections import (
    batch_monitoring,
    datasets,
    drift_analysis,
    drift_impact,
    model_health,
    overview,
    recommendations,
    reports,
)

st.set_page_config(
    page_title="DriftGuard · Banking Drift Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

SECTIONS = {
    "Overview": overview.render,
    "Dataset Management": datasets.render,
    "Customer Drift Analysis": drift_analysis.render,
    "Drift Impact Score": drift_impact.render,
    "Batch Monitoring": batch_monitoring.render,
    "Risk Recommendations": recommendations.render,
    "Model Health": model_health.render,
    "Reports": reports.render,
}


def _sign_in(email: str, password: str) -> None:
    out = safe(lambda: api.login(email, password))
    if out:
        st.session_state.update(
            token=out["access_token"], role=out["role"], name=out["name"], email=email
        )
        api.bump()
        st.rerun()


def sign_in_screen() -> None:
    """Sign-in form; offers first-administrator setup on a fresh installation."""
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
        st.markdown("## 🛡️ DriftGuard")
        st.caption("Banking customer drift monitoring")
        health = api.health()
        if health is None:
            st.error(
                f"The backend at {api.API_ROOT} is not reachable. Start the project with ./start.sh."
            )
            return
        if health.get("status") != "ok":
            st.warning(
                f"Backend degraded — PostgreSQL: {health.get('postgres')}, MongoDB: {health.get('mongo')}"
            )
        status = safe(lambda: api.auth_status(), {}) or {}
        if status.get("has_users") is False:
            st.info("First run: create the administrator account.")
            with st.form("first_admin"):
                name = st.text_input("Full name")
                email = st.text_input("Email")
                password = st.text_input(
                    "Password",
                    type="password",
                    help="At least 8 characters with a letter and a digit.",
                )
                submitted = st.form_submit_button(
                    "Create administrator", type="primary", width="stretch"
                )
                if submitted and safe(
                    lambda: api.register_first_admin(name, email, password)
                ):
                    _sign_in(email, password)
            return
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                _sign_in(email, password)


def sidebar() -> str:
    """Section navigation and session controls."""
    with st.sidebar:
        st.markdown("### 🛡️ DriftGuard")
        st.caption("Banking customer drift monitoring")
        st.divider()
        section = st.radio(
            "Navigation",
            list(SECTIONS),
            key="nav",
            label_visibility="collapsed",
            format_func=lambda s: f"{'ABCDEFGH'[list(SECTIONS).index(s)]}.  {s}",
        )
        st.divider()
        if st.button("↻ Refresh data", width="stretch"):
            api.bump()
            st.rerun()
        st.caption(
            f"Signed in as **{st.session_state.get('name')}**  \n"
            f"{st.session_state.get('email')} · {st.session_state.get('role')}"
        )
        if st.button("Sign out", width="stretch"):
            for key in ("token", "role", "name", "email"):
                st.session_state.pop(key, None)
            st.rerun()
    return section


if not st.session_state.get("token"):
    sign_in_screen()
else:
    SECTIONS[sidebar()]()
