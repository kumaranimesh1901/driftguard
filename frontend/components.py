"""Shared visual language: theme CSS, KPI cards, badges and Plotly helpers."""

from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import ApiError

NAVY = "#0B2545"
BLUE = "#1D5FA8"
SKY = "#8FB8E8"
INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"
BASELINE_COLOUR = "#64748B"
SERIES = ["#2563EB", "#0D9488", "#D97706", "#EA580C", "#7C3AED", "#0284C7", "#DB2777"]

RISK = {
    "LOW RISK": ("#15803D", "#DCFCE7"),
    "MEDIUM RISK": ("#B45309", "#FEF3C7"),
    "HIGH RISK": ("#C2410C", "#FFEDD5"),
    "CRITICAL RISK": ("#B91C1C", "#FEE2E2"),
}
STATUS = {
    "Stable": ("#15803D", "#DCFCE7"),
    "Moderate Drift": ("#B45309", "#FEF3C7"),
    "Significant Drift": ("#B91C1C", "#FEE2E2"),
    "Insufficient data": (MUTED, "#F1F5F9"),
}
PERF = {
    "unknown": ("Unconfirmed — labels pending", MUTED, "#F1F5F9"),
    "unconfirmed": ("Unconfirmed — labels pending", MUTED, "#F1F5F9"),
    "stable": ("Stable (confirmed)", "#15803D", "#DCFCE7"),
    "degraded": ("Degraded (confirmed)", "#C2410C", "#FFEDD5"),
    "severe": ("Severe degradation", "#B91C1C", "#FEE2E2"),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --navy: {NAVY};
  --blue: {BLUE};
  --ink: {INK};
  --muted: {MUTED};
  --line: {LINE};
}}

html, body, [class*="css"] {{
  font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

.stApp {{
  background: #F8FAFC;
}}

.block-container {{
  padding-top: 2.2rem;
  padding-bottom: 3.5rem;
  max-width: 1440px;
}}

/* Sidebar Design */
section[data-testid="stSidebar"] {{
  background: #0B192C !important;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {{
  color: #E2E8F0;
}}

section[data-testid="stSidebar"] .stRadio label p {{
  font-size: 0.92rem;
  font-weight: 500;
  color: #CBD5E1;
  transition: color 0.15s ease;
}}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover p {{
  color: #FFFFFF;
}}

section[data-testid="stSidebar"] hr {{
  border-color: rgba(255, 255, 255, 0.12);
  margin: 1rem 0;
}}

section[data-testid="stSidebar"] button {{
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  color: #F8FAFC !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: all 0.2s ease !important;
}}

section[data-testid="stSidebar"] button:hover {{
  background: rgba(255, 255, 255, 0.14) !important;
  border-color: rgba(255, 255, 255, 0.35) !important;
}}

/* Page Headers */
.dg-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  border-bottom: 1px solid {LINE};
  padding-bottom: 0.9rem;
  margin-bottom: 1.4rem;
}}

.dg-header h1 {{
  font-size: 1.7rem;
  font-weight: 800;
  color: #0F172A;
  margin: 0;
  padding: 0;
  letter-spacing: -0.025em;
}}

.dg-header p {{
  color: #64748B;
  margin: 0.25rem 0 0 0;
  font-size: 0.95rem;
}}

/* KPI Cards */
.dg-kpi {{
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 12px;
  padding: 16px 18px;
  height: 100%;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}}

.dg-kpi:hover {{
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
}}

.dg-kpi .t {{
  color: #64748B;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}}

.dg-kpi .v {{
  color: #0F172A;
  font-size: 1.75rem;
  font-weight: 800;
  line-height: 1.25;
  margin-top: 6px;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}}

.dg-kpi .s {{
  color: #64748B;
  font-size: 0.78rem;
  margin-top: 4px;
}}

/* Badges */
.dg-badge {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 9999px;
  font-weight: 700;
  font-size: 0.75rem;
  letter-spacing: 0.02em;
}}

/* Cards */
.dg-card {{
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  border-radius: 12px;
  padding: 20px 22px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}

.dg-card h4 {{
  margin: 0 0 8px 0;
  color: #0F172A;
  font-size: 1.05rem;
  font-weight: 700;
}}

/* Callouts & Notices */
.dg-note {{
  border-left: 4px solid #2563EB;
  background: #EFF6FF;
  padding: 12px 18px;
  border-radius: 8px;
  color: #1E3A8A;
  font-size: 0.9rem;
  line-height: 1.5;
  margin: 8px 0 16px 0;
}}

.dg-synthetic {{
  border-left: 4px solid #D97706;
  background: #FFFBEB;
  padding: 12px 18px;
  border-radius: 8px;
  color: #92400E;
  font-size: 0.88rem;
  line-height: 1.5;
  margin-bottom: 16px;
}}

/* Risk Banner */
.dg-risk {{
  border-radius: 14px;
  padding: 20px 24px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}

.dg-risk .lbl {{
  font-size: 0.76rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  opacity: 0.88;
}}

.dg-risk .val {{
  font-size: 2rem;
  font-weight: 850;
  letter-spacing: -0.02em;
  margin: 2px 0 6px 0;
}}

/* Form elements & Streamlit refinements */
button[data-baseweb="tab"] {{
  font-size: 0.92rem !important;
  font-weight: 600 !important;
  padding: 8px 16px !important;
}}

button[kind="primary"] {{
  background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
  border: none !important;
  color: #FFFFFF !important;
  box-shadow: 0 2px 6px rgba(37,99,235,0.25) !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  transition: all 0.15s ease !important;
}}

button[kind="primary"]:hover {{
  box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
  transform: translateY(-1px) !important;
}}

div[data-testid="stMetric"] {{
  background: #FFFFFF;
  border: 1px solid #E2E8F0;
  padding: 12px 16px;
  border-radius: 10px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}}

div[data-testid="stMetricValue"] {{
  font-variant-numeric: tabular-nums;
  font-weight: 750;
  color: #0F172A;
}}
</style>
"""


def apply_theme() -> None:
    """Inject the platform stylesheet."""
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str) -> None:
    """Page header."""
    st.markdown(
        f'<div class="dg-header"><div><h1>{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def kpi(col, title: str, value: str, sub: str = "", accent: str | None = None) -> None:
    """KPI card in a column."""
    border = (
        f"border-top:3.5px solid {accent};" if accent else f"border-top:3.5px solid {BLUE};"
    )
    col.markdown(
        f'<div class="dg-kpi" style="{border}"><div class="t">{title}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, fg: str, bg: str, dot: str | None = None) -> str:
    """Inline badge HTML."""
    dot_html = f'<span style="width:6px;height:6px;border-radius:50%;background:{fg};display:inline-block;"></span>' if dot else ""
    return f'<span class="dg-badge" style="color:{fg};background:{bg}">{dot_html}{text}</span>'


def risk_badge(category: str | None) -> str:
    """Badge for a risk category."""
    fg, bg = RISK.get(category or "", (MUTED, "#EEF1F5"))
    return badge(category or "Not analysed", fg, bg, dot=True)


def perf_badge(state: str | None) -> str:
    """Badge for a performance state."""
    label, fg, bg = PERF.get(state or "unknown", PERF["unknown"])
    return badge(label, fg, bg, dot=True)


def note(text: str) -> None:
    """Informational callout."""
    st.markdown(f'<div class="dg-note">💡 {text}</div>', unsafe_allow_html=True)


def synthetic_notice(flag: bool) -> None:
    """Warn that displayed data is synthetic."""
    if flag:
        st.markdown(
            '<div class="dg-synthetic">⚠️ <b>Synthetic demonstration data.</b> Some or all records shown were '
            "generated by DriftGuard's simulator and are not real banking customers.</div>",
            unsafe_allow_html=True,
        )


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
    """Apply the shared chart style."""
    fig.update_layout(
        height=height,
        title=(
            {
                "text": title,
                "font": {"size": 14, "color": NAVY, "family": "Plus Jakarta Sans"},
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
            "l": 12,
            "r": 12,
            "t": (72 if legend else 44) if title else (34 if legend else 12),
            "b": 12,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": INK, "size": 12, "family": "Plus Jakarta Sans"},
        legend=(
            {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.0,
                "xanchor": "left",
                "x": 0,
                "title": None,
                "font": {"size": 11},
            }
            if legend
            else None
        ),
        showlegend=legend,
        hoverlabel={"bgcolor": "white", "font": {"family": "Plus Jakarta Sans"}},
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE)
    fig.update_yaxes(gridcolor="#F1F5F9", zeroline=False)
    return fig


def chart(fig: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure full width."""
    st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})


def series_colour(name: str, index: int) -> str:
    """Baseline is always grey-blue; production series cycle the palette."""
    return (
        BASELINE_COLOUR
        if name.lower().startswith("baseline")
        else SERIES[index % len(SERIES)]
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


def safe(fn: Callable[[], Any], fallback: Any = None) -> Any:
    """Call the API, showing a friendly error instead of a traceback."""
    try:
        return fn()
    except ApiError as exc:
        if exc.code == "unreachable":
            st.error(str(exc))
        elif exc.status == 401:
            st.warning(str(exc))
        else:
            st.error(f"{exc} ({exc.code or exc.status})")
        return fallback


def table(
    df: pd.DataFrame,
    formats: dict[str, str] | None = None,
    styles: dict[str, Callable] | None = None,
    missing: str = "—",
    height: int | None = None,
) -> None:
    """Render a DataFrame with formatted numbers and explicit missing markers.

    ``st.dataframe`` shows nulls as "None"; cells are pre-formatted so missing
    values read as ``missing`` (e.g. "—" or "pending") instead.
    """
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
    fg, bg = STATUS.get(str(val), (INK, "white"))
    return f"color:{fg};background-color:{bg};font-weight:600"


def risk_cell(val: Any) -> str:
    """Pandas Styler colour for a risk category cell."""
    fg, bg = RISK.get(str(val), (INK, "white"))
    return f"color:{fg};background-color:{bg};font-weight:600"
