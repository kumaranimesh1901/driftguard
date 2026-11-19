"""Null-floor normalisation for non-comparable drift measures."""


def normalise(value: float, floor: float, cap: float) -> float:
    """Map an effect size into [0, 1] after its no-drift noise floor."""
    if cap <= floor:
        return float(value > floor)
    return max(0.0, min(1.0, (value - floor) / (cap - floor)))
