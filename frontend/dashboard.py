"""DriftGuard — Credit-Risk Model Monitoring Console.

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
    model_versions,
    overview,
    recommendations,
    reports,
)

st.set_page_config(
    page_title="DriftGuard · Credit Model Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

NAV_GROUPS: dict[str, list[tuple[str, str, str, object]]] = {
    "Monitor": [
        ("Overview", "overview", ":material/dashboard:", overview.render),
        ("Batch Monitoring", "batch-monitoring", ":material/inventory_2:", batch_monitoring.render),
        ("Customer Drift Analysis", "drift-analysis", ":material/query_stats:", drift_analysis.render),
        ("Drift Impact Score", "drift-impact", ":material/bolt:", drift_impact.render),
    ],
    "Model": [
        ("Model Health", "model-health", ":material/monitor_heart:", model_health.render),
        ("Model Versions & Retraining", "model-versions", ":material/history:", model_versions.render),
    ],
    "Act": [
        ("Risk Recommendations", "recommendations", ":material/shield:", recommendations.render),
        ("Reports", "reports", ":material/description:", reports.render),
    ],
    "Data": [
        ("Dataset Management", "datasets", ":material/database:", datasets.render),
    ],
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
    """Centred sign-in screen with quick demo access."""
    _, middle, _ = st.columns([1, 1.35, 1])
    with middle:
        st.markdown("<div style='height:4vh'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 22px;">
                <div style="display: inline-flex; align-items: center; justify-content: center; width: 56px; height: 56px; background: #2563EB; border-radius: 14px; box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25); margin-bottom: 12px; color: #FFFFFF; font-weight: 800; font-size: 24px;">DG</div>
                <h1 style="font-size: 1.85rem; font-weight: 700; color: #0F172A; margin: 0; letter-spacing: -0.025em;">DriftGuard</h1>
                <p style="color: #64748B; font-size: 0.92rem; margin: 6px 0 0 0;">Data-drift detection and model monitoring for credit risk</p>
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
            st.info("Create the initial administrator account to get started.")
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

        with st.container():
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 14px 16px; margin-bottom: 14px;">
                    <div style="font-size: 0.74rem; font-weight: 700; color: #1E40AF; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Quick Demo Access</div>
                    <div style="font-size: 0.84rem; color: #64748B;">Sign in with a pre-configured demonstration profile:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_admin, col_analyst = st.columns(2)
            with col_admin:
                if st.button("Administrator (Full)", key="demo_admin", width="stretch"):
                    _sign_in("admin@example.com", "AdminPassword123")
            with col_analyst:
                if st.button("Risk Analyst (Upload)", key="demo_analyst", width="stretch"):
                    _sign_in("analyst@example.com", "AnalystPassword123")

        st.markdown(
            "<div style='margin: 16px 0 10px 0; text-align: center; color: #94A3B8; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;'>OR SIGN IN WITH CREDENTIALS</div>",
            unsafe_allow_html=True,
        )

        with st.form("login"):
            email = st.text_input("Work Email", placeholder="admin@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••••••")
            if st.form_submit_button("Sign In", type="primary", width="stretch"):
                _sign_in(email, password)

        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px; padding-top: 12px; border-top: 1px solid #E2E8F0; font-size: 0.76rem; color: #64748B;">
                <span style="color: #16A34A; font-weight: 600;">● System Online</span>
                &nbsp;·&nbsp; PostgreSQL: {health.get('postgres', 'connected')}
                &nbsp;·&nbsp; MongoDB: {health.get('mongo', 'connected')}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_sidebar_chrome() -> None:
    """Test-batch toggle, compact user card, live health dots, and actions."""
    with st.sidebar:
        st.divider()

        if "hide_test_batches" not in st.session_state:
            st.session_state["hide_test_batches"] = True
        st.toggle(
            "Hide test batches (UI_Test_*)",
            key="hide_test_batches",
            help="Excludes UI_Test_* batches from Overview, Batch Monitoring and selectors without deleting database records.",
        )

        # Compact User Card
        name = st.session_state.get("name") or "User"
        email = st.session_state.get("email") or ""
        role = st.session_state.get("role") or "reviewer"
        role_colour = (
            "#60A5FA" if role == "administrator" else ("#34D399" if role == "analyst" else "#94A3B8")
        )
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09); border-radius: 10px; padding: 10px 12px; margin: 10px 0 8px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 6px;">
                    <div style="font-size: 0.84rem; font-weight: 600; color: #F8FAFC; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{name}</div>
                    <span style="background: {role_colour}22; color: {role_colour}; border: 1px solid {role_colour}44; font-size: 0.65rem; font-weight: 700; padding: 1px 7px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.04em;">{role}</span>
                </div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{email}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Live 3-dot service status from /health
        h = api.health()
        api_ok = h is not None and h.get("status") == "ok"
        pg_ok = bool(h and h.get("postgres") == "connected")
        mongo_ok = bool(h and h.get("mongo") == "connected")

        def _dot(ok: bool, label: str) -> str:
            colour = "#16A34A" if ok else "#DC2626"
            return (
                f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:0.72rem;color:#CBD5E1;">'
                f'<span style="width:7px;height:7px;border-radius:50%;background:{colour};display:inline-block;"></span>'
                f"{label}</span>"
            )

        st.markdown(
            f"""
            <div style="background: rgba(0,0,0,0.22); border-radius: 8px; padding: 8px 10px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                {_dot(pg_ok, "PostgreSQL")}
                {_dot(mongo_ok, "MongoDB")}
                {_dot(api_ok, "API")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Refresh data", type="secondary", width="stretch", icon=":material/refresh:"):
            api.bump()
            st.rerun()

        if st.button("Log out", type="tertiary", width="stretch", icon=":material/logout:"):
            for key in ("token", "role", "name", "email", "pdf", "selected_job", "selected_batch_id"):
                st.session_state.pop(key, None)
            st.rerun()


def main() -> None:
    if not st.session_state.get("token"):
        login_page = st.Page(sign_in_screen, title="Sign In", url_path="login", default=True)
        pg = st.navigation([login_page], position="hidden")
        pg.run()
        return

    test_target = st.session_state.get("_test_page")
    nav_pages: dict[str, list[st.Page]] = {}
    for group, items in NAV_GROUPS.items():
        nav_pages[group] = [
            st.Page(
                fn,
                title=title,
                url_path=slug,
                icon=icon,
                default=(slug == test_target) if test_target else (slug == "overview"),
            )
            for title, slug, icon, fn in items
        ]

    pg = st.navigation(nav_pages, position="sidebar")
    _render_sidebar_chrome()
    pg.run()


main()
