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
INK = "#1B2433"
MUTED = "#5B6474"
LINE = "#DDE3EC"
BASELINE_COLOUR = "#7C8DA6"
SERIES = ["#1D5FA8", "#2A9D8F", "#E9A23B", "#C2410C", "#6D28D9", "#0E7490", "#BE185D"]

RISK = {
    "LOW RISK": ("#1E7B4F", "#E6F4EC"),
    "MEDIUM RISK": ("#9A6700", "#FFF4D6"),
    "HIGH RISK": ("#C2410C", "#FDECE3"),
    "CRITICAL RISK": ("#B91C1C", "#FDE8E8"),
}
STATUS = {
    "Stable": ("#1E7B4F", "#E6F4EC"),
    "Moderate Drift": ("#9A6700", "#FFF4D6"),
    "Significant Drift": ("#B91C1C", "#FDE8E8"),
    "Insufficient data": (MUTED, "#EEF1F5"),
}
PERF = {
    "unknown": ("Unconfirmed — labels pending", MUTED, "#EEF1F5"),
    "unconfirmed": ("Unconfirmed — labels pending", MUTED, "#EEF1F5"),
    "stable": ("Stable (confirmed)", "#1E7B4F", "#E6F4EC"),
    "degraded": ("Degraded (confirmed)", "#C2410C", "#FDECE3"),
    "severe": ("Severe degradation", "#B91C1C", "#FDE8E8"),
}

CSS = f"""
<style>
:root {{ --navy:{NAVY}; --blue:{BLUE}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; }}
html, body, [class*="css"] {{ font-family: Inter, -apple-system, "Segoe UI", Roboto, sans-serif; }}
.stApp {{ background: #F5F7FA; }}
.block-container {{ padding-top: 3.2rem; max-width: 1400px; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E6ECF5 !important; }}
section[data-testid="stSidebar"] .stRadio label p {{ font-size: 0.92rem; }}
section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.15); }}
section[data-testid="stSidebar"] button {{ background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.25); }}
.dg-header {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid {LINE};
  padding-bottom:0.7rem; margin-bottom:1.1rem; }}
.dg-header h1 {{ font-size:1.55rem; font-weight:700; color:{NAVY}; margin:0; padding:0; }}
.dg-header p {{ color:{MUTED}; margin:0.15rem 0 0 0; font-size:0.9rem; }}
.dg-kpi {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; height:100%;
  box-shadow:0 1px 2px rgba(16,24,40,0.04); }}
.dg-kpi .t {{ color:{MUTED}; font-size:0.72rem; font-weight:600; letter-spacing:0.04em; text-transform:uppercase; }}
.dg-kpi .v {{ color:{INK}; font-size:1.65rem; font-weight:700; line-height:1.25; margin-top:4px;
  font-variant-numeric: tabular-nums; }}
.dg-kpi .s {{ color:{MUTED}; font-size:0.78rem; margin-top:2px; }}
.dg-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:0.78rem; }}
.dg-card {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:16px 18px; margin-bottom:12px; }}
.dg-card h4 {{ margin:0 0 6px 0; color:{NAVY}; font-size:1rem; }}
.dg-note {{ border-left:3px solid {BLUE}; background:#EEF4FB; padding:10px 14px; border-radius:6px; color:{INK};
  font-size:0.88rem; margin:6px 0 12px 0; }}
.dg-synthetic {{ border-left:3px solid #9A6700; background:#FFF8E6; padding:8px 14px; border-radius:6px;
  color:#5C4300; font-size:0.85rem; margin-bottom:12px; }}
.dg-risk {{ border-radius:12px; padding:18px 22px; margin-bottom:12px; }}
.dg-risk .lbl {{ font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.85; }}
.dg-risk .val {{ font-size:1.8rem; font-weight:800; }}
div[data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
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
        f"border-top:3px solid {accent};" if accent else f"border-top:3px solid {BLUE};"
    )
    col.markdown(
        f'<div class="dg-kpi" style="{border}"><div class="t">{title}</div><div class="v">{value}</div>'
        f'<div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def badge(text: str, fg: str, bg: str) -> str:
    """Inline badge HTML."""
    return f'<span class="dg-badge" style="color:{fg};background:{bg}">{text}</span>'


def risk_badge(category: str | None) -> str:
    """Badge for a risk category."""
    fg, bg = RISK.get(category or "", (MUTED, "#EEF1F5"))
    return badge(category or "Not analysed", fg, bg)


def perf_badge(state: str | None) -> str:
    """Badge for a performance state."""
    label, fg, bg = PERF.get(state or "unknown", PERF["unknown"])
    return badge(label, fg, bg)


def note(text: str) -> None:
    """Informational callout."""
    st.markdown(f'<div class="dg-note">{text}</div>', unsafe_allow_html=True)


def synthetic_notice(flag: bool) -> None:
    """Warn that displayed data is synthetic."""
    if flag:
        st.markdown(
            '<div class="dg-synthetic"><b>Synthetic demonstration data.</b> Some or all records shown were '
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
                "font": {"size": 14, "color": NAVY},
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
            "l": 10,
            "r": 10,
            "t": (72 if legend else 44) if title else (34 if legend else 10),
            "b": 10,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": INK, "size": 12},
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
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE)
    fig.update_yaxes(gridcolor="#EEF1F5", zeroline=False)
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
                    fillcolor="rgba(124,141,166,0.18)",
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
