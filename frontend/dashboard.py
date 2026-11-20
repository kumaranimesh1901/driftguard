"""DriftGuard — Banking Customer Drift Monitoring Dashboard.

The user-facing application for credit-risk model surveillance and distribution
drift monitoring. All statistics (PSI, KS, JSD, DIS, model health, recommendations,
reports) are computed by the FastAPI backend.

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
    page_title="DriftGuard · Banking Model Surveillance",
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

SECTION_ICONS = {
    "Overview": "📊",
    "Dataset Management": "📁",
    "Customer Drift Analysis": "📈",
    "Drift Impact Score": "⚡",
    "Batch Monitoring": "📦",
    "Risk Recommendations": "🛡️",
    "Model Health": "🩺",
    "Reports": "📄",
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
    """Polished modern sign-in screen with quick 1-click demo access."""
    _, middle, _ = st.columns([1, 1.4, 1])
    with middle:
        st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 24px;">
                <div style="display: inline-flex; align-items: center; justify-content: center; width: 68px; height: 68px; background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%); border-radius: 18px; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28); margin-bottom: 14px;">
                    <span style="font-size: 34px;">🛡️</span>
                </div>
                <h1 style="font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.03em;">DriftGuard</h1>
                <p style="color: #64748B; font-size: 0.95rem; margin: 6px 0 0 0;">Banking Customer & Credit Model Surveillance Platform</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        health = api.health()
        if health is None:
            st.error(
                f"The backend at `{api.API_ROOT}` is not reachable. Start the services with `./start.sh`."
            )
            return
        if health.get("status") != "ok":
            st.warning(
                f"Backend degraded — PostgreSQL: {health.get('postgres')}, MongoDB: {health.get('mongo')}"
            )

        status = safe(lambda: api.auth_status(), {}) or {}
        if status.get("has_users") is False:
            st.info("👋 Welcome! Create the initial administrator account to get started.")
            with st.form("first_admin"):
                name = st.text_input("Full name", placeholder="e.g. Jane Doe")
                email = st.text_input("Work Email", placeholder="e.g. admin@bank.com")
                password = st.text_input(
                    "Password",
                    type="password",
                    help="At least 8 characters with at least one letter and one digit.",
                )
                submitted = st.form_submit_button(
                    "Create Administrator Account", type="primary", width="stretch"
                )
                if submitted and safe(
                    lambda: api.register_first_admin(name, email, password)
                ):
                    _sign_in(email, password)
            return

        # 1-Click Demo Section
        with st.container():
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px 18px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.04);">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">⚡ Quick Demo Access</div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 12px;">Sign in instantly using pre-configured test profiles:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_admin, col_analyst = st.columns(2)
            with col_admin:
                if st.button("👑 Administrator\n(Full Platform)", key="demo_admin", width="stretch"):
                    _sign_in("admin@example.com", "AdminPassword123")
            with col_analyst:
                if st.button("📊 Risk Analyst\n(Surveillance & Upload)", key="demo_analyst", width="stretch"):
                    _sign_in("analyst@example.com", "AnalystPassword123")

        st.markdown(
            "<div style='margin: 18px 0 12px 0; text-align: center; color: #94A3B8; font-size: 0.82rem; font-weight: 600;'>— OR ENTER YOUR CREDENTIALS —</div>",
            unsafe_allow_html=True,
        )

        with st.form("login"):
            email = st.text_input("Work Email", placeholder="admin@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••••••")
            if st.form_submit_button("Sign In to DriftGuard", type="primary", width="stretch"):
                _sign_in(email, password)

        # Telemetry badge
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 24px; padding-top: 14px; border-top: 1px solid #E2E8F0; font-size: 0.78rem; color: #64748B;">
                <span style="color: #16A34A; font-weight: 700;">● System Online</span> &nbsp;·&nbsp; PostgreSQL: {health.get('postgres', 'connected')} &nbsp;·&nbsp; MongoDB: {health.get('mongo', 'connected')}
            </div>
            """,
            unsafe_allow_html=True,
        )


def sidebar() -> str:
    """Rich sidebar navigation and session telemetry."""
    with st.sidebar:
        # Branding
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0 14px 0;">
                <div style="display: flex; align-items: center; justify-content: center; width: 42px; height: 42px; background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%); border-radius: 12px; font-size: 22px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);">🛡️</div>
                <div>
                    <div style="font-weight: 800; font-size: 1.2rem; color: #FFFFFF; letter-spacing: -0.02em;">DriftGuard</div>
                    <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">Credit Surveillance v0.1</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # User Card
        name = st.session_state.get("name") or "User"
        email = st.session_state.get("email") or ""
        role = st.session_state.get("role") or "reviewer"
        role_badge_color = (
            "#3B82F6" if role == "administrator" else ("#10B981" if role == "analyst" else "#94A3B8")
        )

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px 14px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #FFFFFF;">👤 {name}</div>
                    <span style="background: {role_badge_color}22; color: {role_badge_color}; border: 1px solid {role_badge_color}55; font-size: 0.68rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.04em;">{role}</span>
                </div>
                <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin: 12px 0 6px 0;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        section = st.radio(
            "Navigation",
            list(SECTIONS),
            key="nav",
            label_visibility="collapsed",
            format_func=lambda s: f"{SECTION_ICONS.get(s, '•')}  {s}",
        )

        st.divider()

        # Action Buttons
        if st.button("↻  Refresh Surveillance Data", width="stretch"):
            api.bump()
            st.rerun()

        if st.button("Log Out", width="stretch"):
            for key in ("token", "role", "name", "email", "pdf", "selected_job"):
                st.session_state.pop(key, None)
            st.rerun()

        # Telemetry footer
        st.markdown(
            """
            <div style="background: rgba(0,0,0,0.25); border-radius: 8px; padding: 10px 12px; margin-top: 14px; font-size: 0.72rem; color: #94A3B8;">
                <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background: #10B981; display: inline-block;"></span>
                    <span style="color: #E2E8F0; font-weight: 600;">System Online</span>
                </div>
                <div>PostgreSQL · MongoDB · FastAPI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return section


if not st.session_state.get("token"):
    sign_in_screen()
else:
    SECTIONS[sidebar()]()
