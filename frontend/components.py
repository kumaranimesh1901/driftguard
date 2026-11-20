"""Reusable UI components built on frontend/theme.py."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import api_client as api
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import theme
from api_client import ApiError

# Backwards-compatible exports for existing imports
NAVY = theme.DIS_COLOR
BLUE = theme.PRIMARY
SKY = "#93C5FD"
INK = theme.INK
MUTED = theme.MUTED
LINE = theme.BORDER
BASELINE_COLOUR = theme.BASELINE_COLOR
SERIES = theme.SERIES_COLORS
RISK = {k: v for k, v in theme.RISK_COLORS.items() if k != "UNKNOWN"}

STATUS = {
    "Stable": theme.SEVERITY_COLORS["low"],
    "Moderate Drift": theme.SEVERITY_COLORS["moderate"],
    "Significant Drift": theme.SEVERITY_COLORS["critical"],
    "Insufficient data": theme.SEVERITY_COLORS["unknown"],
}
PERF = {
    "unknown": ("Unknown – labels pending", *theme.TONES["unknown"]),
    "unconfirmed": ("Unknown – labels pending", *theme.TONES["unknown"]),
    "stable": ("Stable (confirmed)", *theme.TONES["low"]),
    "degraded": ("Degraded (confirmed)", *theme.TONES["high"]),
    "severe": ("Severe degradation", *theme.TONES["critical"]),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --primary: {theme.PRIMARY};
  --bg: {theme.BG};
  --surface: {theme.SURFACE};
  --ink: {theme.INK};
  --text: {theme.TEXT};
  --muted: {theme.MUTED};
  --border: {theme.BORDER};
}}

html, body, [class*="css"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: {theme.INK};
}}

.stApp {{
  background: {theme.BG};
}}

/* Hide Streamlit Deploy button and chrome while keeping sidebar collapse working */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu,
footer {{
  display: none !important;
}}

header[data-testid="stHeader"] {{
  background: transparent !important;
  height: 2.5rem;
}}

.block-container {{
  padding-top: 2.5rem !important;
  padding-bottom: 3rem !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  max-width: 1440px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}}

/* Equal-height card rows in Streamlit horizontal blocks */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
  display: flex;
  flex-direction: column;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: #0F172A !important;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  padding-top: 0.5rem;
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {{
  color: #E2E8F0;
}}

/* Brand banner at top of sidebar */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]::after {{
  content: "DriftGuard";
  font-weight: 700;
  font-size: 1.05rem;
  color: #F8FAFC;
  letter-spacing: -0.02em;
  margin-left: 4px;
}}

section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {{
  font-size: 0.7rem !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: #64748B !important;
  padding-top: 0.65rem !important;
  padding-bottom: 0.2rem !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{
  border-radius: 8px !important;
  padding: 0.36rem 0.65rem !important;
  margin: 1px 0 !important;
  color: #CBD5E1 !important;
  transition: background 0.15s ease, color 0.15s ease;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover {{
  background: rgba(255, 255, 255, 0.08) !important;
  color: #FFFFFF !important;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: rgba(37, 99, 235, 0.25) !important;
  color: #FFFFFF !important;
  border-left: 3px solid #3B82F6;
}}

section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span {{
  color: #FFFFFF !important;
  font-weight: 600 !important;
}}

section[data-testid="stSidebar"] hr {{
  border-color: rgba(255, 255, 255, 0.10);
  margin: 0.65rem 0;
}}

section[data-testid="stSidebar"] button[kind="secondary"] {{
  background: rgba(255, 255, 255, 0.07) !important;
  border: 1px solid rgba(255, 255, 255, 0.16) !important;
  color: #F1F5F9 !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
  font-size: 0.84rem !important;
}}

section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
  background: rgba(255, 255, 255, 0.14) !important;
}}

section[data-testid="stSidebar"] button[kind="tertiary"] {{
  background: transparent !important;
  border: none !important;
  color: #94A3B8 !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
}}

section[data-testid="stSidebar"] button[kind="tertiary"]:hover {{
  color: #F8FAFC !important;
  background: rgba(255, 255, 255, 0.06) !important;
}}

/* Typography & Headers */
.dg-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 16px;
  border-bottom: 1px solid {theme.BORDER};
  padding-bottom: 12px;
  margin-bottom: 20px;
}}

.dg-header h1 {{
  font-size: 28px;
  font-weight: 600;
  color: {theme.INK};
  margin: 0;
  line-height: 1.25;
  letter-spacing: -0.02em;
}}

.dg-header p {{
  color: {theme.MUTED};
  margin: 4px 0 0 0;
  font-size: 14px;
  line-height: 1.45;
}}

.dg-header-meta {{
  font-size: 12px;
  color: {theme.MUTED};
  text-align: right;
  margin-top: 2px;
  font-variant-numeric: tabular-nums;
}}

.dg-section {{
  margin: 20px 0 12px 0;
}}

.dg-section h2 {{
  font-size: 18px;
  font-weight: 600;
  color: {theme.INK};
  margin: 0;
  line-height: 1.3;
}}

.dg-section p {{
  font-size: 12px;
  color: {theme.MUTED};
  margin: 2px 0 0 0;
}}

/* KPI Cards */
.dg-kpi {{
  background: {theme.SURFACE};
  border: 1px solid {theme.BORDER};
  border-radius: 12px;
  padding: 16px 18px;
  height: 100%;
  min-height: 112px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}

.dg-kpi .t {{
  color: {theme.MUTED};
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}

.dg-kpi .v {{
  color: {theme.INK};
  font-size: 28px;
  font-weight: 700;
  line-height: 1.15;
  margin-top: 6px;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}}

.dg-kpi .s {{
  color: {theme.MUTED};
  font-size: 12px;
  margin-top: 6px;
  line-height: 1.35;
}}

/* Badges & Pills */
.dg-badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 9999px;
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.02em;
  white-space: nowrap;
}}

.dg-chip {{
  display: inline-block;
  background: #F1F5F9;
  border: 1px solid #CBD5E1;
  color: {theme.INK};
  padding: 2px 9px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  margin: 2px 4px 2px 0;
}}

/* Surface Cards */
.dg-card {{
  background: {theme.SURFACE};
  border: 1px solid {theme.BORDER};
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 16px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}}

.dg-card h4 {{
  margin: 0 0 8px 0;
  color: {theme.INK};
  font-size: 15px;
  font-weight: 600;
}}

/* Callouts */
.dg-callout {{
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 13px;
  line-height: 1.5;
  margin: 8px 0 16px 0;
}}

/* Evidence Row */
.dg-ev-row {{
  display: grid;
  grid-template-columns: 1.5fr 2.2fr 64px;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid #F1F5F9;
  font-size: 13px;
}}

.dg-ev-row:last-child {{
  border-bottom: none;
}}

.dg-ev-bar-track {{
  height: 6px;
  background: #F1F5F9;
  border-radius: 999px;
  overflow: hidden;
  flex: 1;
}}

/* Empty State */
.dg-empty {{
  background: {theme.SURFACE};
  border: 1px dashed #CBD5E1;
  border-radius: 12px;
  padding: 32px 24px;
  text-align: center;
  margin: 12px 0 16px 0;
}}

.dg-empty .title {{
  font-size: 15px;
  font-weight: 600;
  color: {theme.INK};
  margin-top: 6px;
}}

.dg-empty .hint {{
  font-size: 13px;
  color: {theme.MUTED};
  margin-top: 4px;
  max-width: 480px;
  margin-left: auto;
  margin-right: auto;
}}

/* Buttons & Tabs */
button[data-baseweb="tab"] {{
  font-size: 14px !important;
  font-weight: 600 !important;
  padding: 8px 14px !important;
}}

button[kind="primary"] {{
  background: {theme.PRIMARY} !important;
  border: 1px solid #1D4ED8 !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
}}

button[kind="primary"]:hover {{
  background: #1D4ED8 !important;
}}

div[data-testid="stMetric"] {{
  background: {theme.SURFACE};
  border: 1px solid {theme.BORDER};
  padding: 12px 16px;
  border-radius: 10px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}}

div[data-testid="stMetricValue"] {{
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  color: {theme.INK};
}}
</style>
"""


def apply_theme() -> None:
    """Inject the platform stylesheet."""
    st.markdown(CSS, unsafe_allow_html=True)


def format_date(value: str | None) -> str:
    """Format an ISO date string like '2026-11-15' as '15 Nov 2026'."""
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%d %b %Y").lstrip("0")
    except ValueError:
        return str(value)


def format_timestamp(value: str | None) -> str:
    """Format an ISO timestamp in local time, e.g. '21 Nov 2026, 00:15'."""
    if not value:
        return "Not analysed"
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%d %b %Y, %H:%M").lstrip("0")
    except ValueError:
        return str(value)


def format_perf_state(state: str | None) -> str:
    """Standard display label for performance states; never blank or 'No'."""
    if not state or str(state).lower() in ("unknown", "unconfirmed", "none"):
        return "Unknown – labels pending"
    return PERF.get(str(state).lower(), (str(state).title(), theme.MUTED, "#F1F5F9"))[0]


def header(title: str, subtitle: str) -> None:
    """Simple page header (backwards-compatible wrapper)."""
    st.markdown(
        f'<div class="dg-header"><div><h1>{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def section(title: str, caption: str | None = None) -> None:
    """Consistent section heading (18px/600) with optional 12px caption."""
    cap = f"<p>{caption}</p>" if caption else ""
    st.markdown(f'<div class="dg-section"><h2>{title}</h2>{cap}</div>', unsafe_allow_html=True)


def status_pill(text: str, tone: str = "neutral", dot: bool = True, tooltip: str = "") -> str:
    """Semantic status pill HTML."""
    fg, bg = theme.TONES.get(tone.lower(), theme.TONES["neutral"])
    dot_html = (
        f'<span style="width:6px;height:6px;border-radius:50%;background:{fg};display:inline-block;"></span>'
        if dot
        else ""
    )
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<span class="dg-badge"{title_attr} style="color:{fg};background:{bg};border:1px solid {fg}2E;">'
        f"{dot_html}{text}</span>"
    )


def badge(text: str, fg: str, bg: str, dot: bool = False, tooltip: str = "") -> str:
    """Inline badge HTML."""
    dot_html = (
        f'<span style="width:6px;height:6px;border-radius:50%;background:{fg};display:inline-block;"></span>'
        if dot
        else ""
    )
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<span class="dg-badge"{title_attr} style="color:{fg};background:{bg};border:1px solid {fg}2E;">'
        f"{dot_html}{text}</span>"
    )


def risk_badge(category: str | None) -> str:
    """Semantic risk badge."""
    fg, bg = theme.risk_color(category)
    return badge(category or "Not analysed", fg, bg, dot=True)


def perf_badge(state: str | None) -> str:
    """Semantic performance state badge."""
    key = (state or "unknown").lower()
    label, fg, bg = PERF.get(key, PERF["unknown"])
    return badge(label, fg, bg, dot=True)


def info_callout(kind: str, text: str) -> None:
    """Callout banner: kind in {'info', 'warning', 'success', 'danger'}."""
    fg, bg = theme.TONES.get(kind, theme.TONES["info"])
    text_color = {
        "info": "#1E3A8A",
        "warning": "#92400E",
        "success": "#166534",
        "danger": "#991B1B",
    }.get(kind, theme.INK)
    st.markdown(
        f'<div class="dg-callout" style="background:{bg};border:1px solid {fg}33;border-left:3.5px solid {fg};color:{text_color};">{text}</div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Informational callout (wrapper over info_callout)."""
    info_callout("info", text)


def synthetic_notice(flag: bool) -> None:
    """Compact synthetic-data callout."""
    if flag:
        info_callout(
            "warning",
            "<b>Synthetic demonstration data.</b> Records shown were generated by DriftGuard's simulator and are not real banking customers.",
        )


def empty_state(
    icon: str,
    title: str,
    hint: str,
    action_label: str | None = None,
    action_fn: Callable[[], None] | None = None,
    key: str | None = None,
) -> None:
    """Consistent empty state card with optional action button."""
    st.markdown(
        f"""
        <div class="dg-empty">
            <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:{theme.MUTED};">{icon}</div>
            <div class="title">{title}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_label and action_fn:
        _, mid, _ = st.columns([2, 1.5, 2])
        if mid.button(action_label, type="primary", width="stretch", key=key or f"empty_{title}"):
            action_fn()


def kpi_card(
    col,
    label: str,
    value: str,
    sublabel: str = "",
    tone: str = "primary",
    delta: str | None = None,
    help: str | None = None,
) -> None:
    """Equal-height KPI card following the Phase 3 design system."""
    fg, _ = theme.TONES.get(tone.lower(), (theme.PRIMARY, "#EFF6FF"))
    title_attr = f' title="{help}"' if help else ""
    help_mark = (
        f' <span style="color:#94A3B8;cursor:help;" title="{help}">ⓘ</span>'
        if help
        else ""
    )
    delta_html = (
        f'<span style="font-size:12px;font-weight:600;color:{fg};margin-left:6px;">{delta}</span>'
        if delta
        else ""
    )
    col.markdown(
        f'<div class="dg-kpi" style="border-top:3px solid {fg};"{title_attr}>'
        f'<div class="t">{label}{help_mark}</div>'
        f'<div class="v">{value}{delta_html}</div>'
        f'<div class="s">{sublabel}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def kpi(col, title: str, value: str, sub: str = "", accent: str | None = None) -> None:
    """Backwards-compatible KPI card accepting a hex accent."""
    color = accent or theme.PRIMARY
    col.markdown(
        f'<div class="dg-kpi" style="border-top:3px solid {color};">'
        f'<div class="t">{title}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def evidence_row(
    feature: str, d: float, fi: float, s: float, dis: float, severity: str | None = None
) -> str:
    """Return HTML for a single DIS evidence row with D/FI/S mini bars."""
    def _bar(label: str, val: float, color: str) -> str:
        pct = max(0.0, min(100.0, float(val or 0.0) * 100.0))
        return (
            f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
            f'<span style="width:18px;font-size:10px;font-weight:600;color:{theme.MUTED};">{label}</span>'
            f'<div class="dg-ev-bar-track"><div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:999px;"></div></div>'
            f'<span style="width:32px;text-align:right;font-size:11px;color:{theme.TEXT};font-variant-numeric:tabular-nums;">{val:.2f}</span>'
            f"</div>"
        )

    sev_fg, _ = theme.severity_color(severity)
    dot = (
        f'<span style="width:7px;height:7px;border-radius:50%;background:{sev_fg};display:inline-block;margin-right:6px;"></span>'
        if severity
        else ""
    )
    return (
        f'<div class="dg-ev-row">'
        f'<div style="font-weight:600;color:{theme.INK};">{dot}{feature}</div>'
        f'<div>{_bar("D", d, theme.D_COLOR)}{_bar("FI", fi, theme.FI_COLOR)}{_bar("S", s, theme.S_COLOR)}</div>'
        f'<div style="text-align:right;font-weight:700;font-size:14px;color:{theme.DIS_COLOR};font-variant-numeric:tabular-nums;">{dis:.3f}</div>'
        f"</div>"
    )


def page_header(
    title: str,
    subtitle: str,
    *,
    show_batch_selector: bool = True,
    badge_html: str = "",
) -> dict[str, Any] | None:
    """Global page header with title, subtitle, and synced batch selector on the right."""
    from sections.shared import latest_batch, visible_batches

    selected_meta: dict[str, Any] | None = None
    batches = visible_batches(safe(lambda: api.get("/batches"), []) or []) if show_batch_selector else []

    if show_batch_selector and batches:
        left, right = st.columns([3.2, 1.4], vertical_alignment="bottom")
        with left:
            pill = f"&nbsp;&nbsp;{badge_html}" if badge_html else ""
            st.markdown(
                f'<div style="padding-bottom:4px;"><h1 style="font-size:28px;font-weight:600;color:{theme.INK};margin:0;line-height:1.25;letter-spacing:-0.02em;">{title}{pill}</h1>'
                f'<p style="color:{theme.MUTED};margin:4px 0 0 0;font-size:14px;">{subtitle}</p></div>',
                unsafe_allow_html=True,
            )
        with right:
            ordered = sorted(
                batches,
                key=lambda b: (b.get("batch_date") or "", b.get("uploaded_at") or "", b["batch_id"]),
                reverse=True,
            )
            ids = [b["batch_id"] for b in ordered]
            by_id = {b["batch_id"]: b for b in ordered}
            default_batch = latest_batch(batches)
            current = st.session_state.get("selected_batch_id")
            if current not in ids:
                current = default_batch["batch_id"] if default_batch else ids[0]
                st.session_state["selected_batch_id"] = current

            def _fmt_batch(bid: str) -> str:
                b = by_id[bid]
                la = b.get("latest_analysis") or {}
                risk = (la.get("risk_category") or "Not analysed").replace(" RISK", "")
                return f"{bid} ({format_date(b.get('batch_date'))}) · {risk}"

            picked = st.selectbox(
                "Active batch",
                ids,
                index=ids.index(current),
                format_func=_fmt_batch,
                key=f"hdr_batch_{title.lower().replace(' ', '_')}",
                label_visibility="collapsed",
            )
            if picked != st.session_state.get("selected_batch_id"):
                st.session_state["selected_batch_id"] = picked
                if by_id[picked].get("latest_job_id"):
                    st.session_state["selected_job"] = by_id[picked]["latest_job_id"]
                st.rerun()
            selected_meta = by_id[picked]
            if selected_meta.get("latest_job_id") and st.session_state.get("selected_job") is None:
                st.session_state["selected_job"] = selected_meta["latest_job_id"]
            la = selected_meta.get("latest_analysis") or {}
            st.markdown(
                f'<div class="dg-header-meta">Last analysed: {format_timestamp(la.get("analysed_at"))}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="border-bottom:1px solid {theme.BORDER};margin:6px 0 18px 0;"></div>',
            unsafe_allow_html=True,
        )
    else:
        pill = f"&nbsp;&nbsp;{badge_html}" if badge_html else ""
        st.markdown(
            f'<div class="dg-header"><div><h1>{title}{pill}</h1><p>{subtitle}</p></div></div>',
            unsafe_allow_html=True,
        )
    return selected_meta


def fmt(value: Any, spec: str = ".3f", none: str = "—") -> str:
    """Format a number or return a dash."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return none
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def layout(
    fig: go.Figure, height: int = 340, title: str | None = None, legend: bool = True
) -> go.Figure:
    """Apply the shared Plotly template from theme.py."""
    fig.update_layout(
        template=theme.plotly_template(),
        height=height,
        title=(
            {
                "text": title,
                "font": {"size": 14, "color": theme.INK},
                "x": 0,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
                "yref": "container",
            }
            if title
            else None
        ),
        margin={
            "l": 56,
            "r": 24,
            "t": (72 if legend else 48) if title else (40 if legend else 24),
            "b": 48,
        },
        showlegend=legend,
    )
    return fig


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure full width with mode bar hidden."""
    st.plotly_chart(fig, width="stretch", key=key, config=theme.PLOTLY_CONFIG)


def series_colour(name: str, index: int) -> str:
    """Baseline is always neutral slate; production series cycle the palette."""
    return (
        theme.BASELINE_COLOR
        if name.lower().startswith("baseline")
        else theme.SERIES_COLORS[index % len(theme.SERIES_COLORS)]
    )


def distribution_figure(
    dist: dict[str, Any],
    title: str | None = None,
    height: int = 320,
    names: dict[str, str] | None = None,
    legend: bool = True,
) -> go.Figure:
    """Overlaid distributions from backend histogram payloads."""
    fig = go.Figure()
    names = names or {}
    if not dist or not dist.get("series"):
        return layout(fig, height, title)
    if dist["kind"] == "numerical":
        edges = dist["edges"]
        centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        for i, (name, props) in enumerate(dist["series"].items()):
            label = names.get(name, name)
            colour = series_colour(label, i)
            fig.add_trace(
                go.Scatter(
                    x=centres,
                    y=props,
                    name=label,
                    mode="lines",
                    line={"color": colour, "width": 2.2},
                    fill="tozeroy" if label.lower().startswith("baseline") else None,
                    fillcolor="rgba(100,116,139,0.14)",
                    hovertemplate=f"{label}<br>%{{x:,.2f}}: %{{y:.1%}}<extra></extra>",
                )
            )
    else:
        for i, (name, props) in enumerate(dist["series"].items()):
            label = names.get(name, name)
            fig.add_trace(
                go.Bar(
                    x=dist["categories"],
                    y=props,
                    name=label,
                    marker_color=series_colour(label, i),
                    hovertemplate=f"{label}<br>%{{x}}: %{{y:.1%}}<extra></extra>",
                )
            )
        fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(tickformat=".0%", title="Share of customers")
    return layout(fig, height, title, legend=legend)


def safe(fn: Callable[[], Any], fallback: Any = None, retry_key: str | None = None) -> Any:
    """Call the API, showing a friendly error card with Retry instead of a traceback."""
    try:
        return fn()
    except ApiError as exc:
        kind = "warning" if exc.status == 401 else "danger"
        detail = str(exc) if exc.code == "unreachable" or exc.status == 401 else f"{exc} ({exc.code or exc.status})"
        info_callout(kind, f"<b>Unable to load data.</b> {detail}")
        if retry_key and st.button("Retry", key=f"retry_{retry_key}", type="secondary"):
            api.bump()
            st.rerun()
        return fallback


def table(
    df: pd.DataFrame,
    formats: dict[str, str] | None = None,
    styles: dict[str, Callable] | None = None,
    missing: str = "—",
    height: int | None = None,
) -> None:
    """Render a DataFrame with formatted numbers and explicit missing markers."""
    out = df.copy()
    for col, spec in (formats or {}).items():
        if col in out.columns:
            out[col] = [
                (
                    missing
                    if v is None or (isinstance(v, float) and pd.isna(v))
                    else format(v, spec)
                )
                for v in out[col]
            ]
    out = out.astype(object).where(out.notna(), missing)
    styled = out.style
    for col, fn in (styles or {}).items():
        if col in out.columns:
            styled = styled.map(fn, subset=[col])
    kwargs = {"height": height} if height else {}
    st.dataframe(styled, hide_index=True, width="stretch", **kwargs)


def status_cell(val: Any) -> str:
    """Pandas Styler colour for a drift status cell."""
    fg, bg = STATUS.get(str(val), (theme.INK, "white"))
    return f"color:{fg};background-color:{bg};font-weight:600"


def risk_cell(val: Any) -> str:
    """Pandas Styler colour for a risk category cell."""
    fg, bg = theme.risk_color(str(val))
    return f"color:{fg};background-color:{bg};font-weight:600"
