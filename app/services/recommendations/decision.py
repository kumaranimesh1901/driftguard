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
        base = (
            "Investigate affected features and data quality"
            if level == "high"
            else "Investigate and monitor closely"
        )
        # Labels already confirmed stable performance; only ask for them when unknown.
        suffix = "" if performance == "stable" else "; labels needed to confirm impact"
        return "drift-warning", level, base + suffix
    return None, "low", "Continue monitoring"
