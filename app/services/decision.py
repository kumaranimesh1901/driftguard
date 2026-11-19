"""Explicit decision engine that keeps warning and degradation distinct."""


def decide(level: str, performance: str) -> tuple[str | None, str, str]:
    """Return alert type, level and non-fabricated baseline recommendation."""
    if performance == "severe":
        return (
            "confirmed-degradation",
            "high",
            "Immediate model review and controlled retraining",
        )
    if performance == "degraded":
        return (
            "confirmed-degradation",
            "high" if level == "high" else "moderate",
            (
                "Consider retraining"
                if level != "low"
                else "Investigate possible concept drift or label/pipeline issues (performance dropped without strong input drift)"
            ),
        )
    if level in {"moderate", "high"}:
        return (
            "drift-warning",
            level,
            (
                "Investigate affected features and data quality; labels needed to confirm impact"
                if level == "high"
                else "Investigate and monitor closely; labels needed to confirm impact"
            ),
        )
    return None, "low", "Continue monitoring"
