"""Null-floor normalisation for non-comparable drift measures."""

from app.core.config import settings


def normalise(value: float, floor: float, cap: float) -> float:
    """Map an effect size into [0, 1] after its no-drift noise floor."""
    if cap <= floor:
        return float(value > floor)
    return max(0.0, min(1.0, (value - floor) / (cap - floor)))


def normalise_psi(
    value: float, floor: float = 0.05, cap: float = settings.norm_cap_psi
) -> float:
    """Map PSI value into [0, 1]."""
    return normalise(value, floor, cap)


def normalise_ks(
    value: float, floor: float = 0.05, cap: float = settings.norm_cap_ks
) -> float:
    """Map KS statistic into [0, 1]."""
    return normalise(value, floor, cap)


def normalise_jsd(
    value: float, floor: float = 0.02, cap: float = settings.norm_cap_jsd
) -> float:
    """Map JSD divergence into [0, 1]."""
    return normalise(value, floor, cap)


def aggregate_d(d_values: list[float], method: str = settings.d_aggregation) -> float:
    """Aggregate normalised detector values."""
    if not d_values:
        return 0.0
    if method == "mean":
        return float(sum(d_values) / len(d_values))
    return float(max(d_values))
