"""Single source of truth for DriftGuard visual tokens and Plotly template."""

from typing import Any

import plotly.graph_objects as go

# Core palette
PRIMARY = "#2563EB"
BG = "#F7F8FA"
SURFACE = "#FFFFFF"
INK = "#0F172A"
TEXT = "#334155"
MUTED = "#64748B"
BORDER = "#E2E8F0"
GRID = "#EEF2F6"
BASELINE_COLOR = "#64748B"

# Semantic risk & severity colours: (foreground, background)
RISK_COLORS: dict[str, tuple[str, str]] = {
    "LOW RISK": ("#16A34A", "#F0FDF4"),
    "MEDIUM RISK": ("#D97706", "#FFFBEB"),
    "HIGH RISK": ("#EA580C", "#FFF7ED"),
    "CRITICAL RISK": ("#DC2626", "#FEF2F2"),
    "UNKNOWN": ("#64748B", "#F1F5F9"),
}

SEVERITY_COLORS: dict[str, tuple[str, str]] = {
    "low": ("#16A34A", "#F0FDF4"),
    "moderate": ("#D97706", "#FFFBEB"),
    "high": ("#EA580C", "#FFF7ED"),
    "critical": ("#DC2626", "#FEF2F2"),
    "unknown": ("#64748B", "#F1F5F9"),
}

TONES: dict[str, tuple[str, str]] = {
    "low": ("#16A34A", "#F0FDF4"),
    "success": ("#16A34A", "#F0FDF4"),
    "medium": ("#D97706", "#FFFBEB"),
    "warning": ("#D97706", "#FFFBEB"),
    "high": ("#EA580C", "#FFF7ED"),
    "critical": ("#DC2626", "#FEF2F2"),
    "danger": ("#DC2626", "#FEF2F2"),
    "info": ("#2563EB", "#EFF6FF"),
    "primary": ("#2563EB", "#EFF6FF"),
    "neutral": ("#64748B", "#F1F5F9"),
    "unknown": ("#64748B", "#F1F5F9"),
}

# Categorical palette for DIS decomposition components
D_COLOR = "#2563EB"
FI_COLOR = "#7C3AED"
S_COLOR = "#0891B2"
DIS_COLOR = "#0F172A"

SERIES_COLORS = [
    "#2563EB",
    "#0891B2",
    "#7C3AED",
    "#D97706",
    "#EA580C",
    "#16A34A",
    "#DB2777",
]

# Technical tooltips used consistently across all pages
HELP_TEXTS: dict[str, str] = {
    "psi": "Population Stability Index: measures binned shift between baseline and production (>0.25 is conventionally flagged).",
    "ks": "Kolmogorov–Smirnov statistic: maximum vertical distance between baseline and production empirical CDFs (numerical features).",
    "jsd": "Jensen–Shannon Divergence (base 2, bounded in [0, 1]): symmetric distribution distance.",
    "noise_floor": "95th-percentile sampling noise floor estimated via bootstrap from the baseline under no drift.",
    "d": "Normalised Drift Magnitude D ∈ [0, 1]: excess above the bootstrap noise floor, aggregated across detectors.",
    "fi": "Feature Importance FI ∈ [0, 1]: mean absolute SHAP attribution of the champion credit model, normalised to max=1.",
    "s": "Model Sensitivity S ∈ [0, 1]: mean prediction change when permuting the feature, normalised to max=1.",
    "dis": "Drift Impact Score = D × FI × S ∈ [0, 1]: prioritises drift by estimated impact on credit-model predictions.",
    "adwin": "ADWIN (Adaptive Windowing, River): streaming change detector on predicted-probability and |y − p| error streams.",
}

PLOTLY_CONFIG: dict[str, Any] = {
    "displayModeBar": False,
    "displaylogo": False,
}


def plotly_template() -> go.layout.Template:
    """Shared Plotly layout template used by every chart."""
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font={"family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif", "size": 12, "color": INK},
            margin={"l": 56, "r": 24, "t": 48, "b": 48},
            hovermode="x unified",
            hoverlabel={"bgcolor": SURFACE, "bordercolor": BORDER, "font": {"size": 12, "color": INK}},
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
                "title": None,
                "font": {"size": 11, "color": TEXT},
            },
            xaxis={
                "showgrid": False,
                "linecolor": BORDER,
                "ticks": "outside",
                "tickcolor": BORDER,
                "automargin": True,
                "title": {"font": {"size": 12, "color": MUTED}},
            },
            yaxis={
                "gridcolor": GRID,
                "zeroline": False,
                "automargin": True,
                "title": {"font": {"size": 12, "color": MUTED}},
            },
        )
    )


def risk_color(category: str | None) -> tuple[str, str]:
    """Return (fg, bg) for a risk category string."""
    return RISK_COLORS.get((category or "").upper(), RISK_COLORS["UNKNOWN"])


def severity_color(severity: str | None) -> tuple[str, str]:
    """Return (fg, bg) for a feature severity level."""
    return SEVERITY_COLORS.get((severity or "unknown").lower(), SEVERITY_COLORS["unknown"])
