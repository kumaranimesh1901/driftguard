"""Kolmogorov-Smirnov detector."""

import numpy as np
from scipy.stats import ks_2samp


def ks(
    reference: np.ndarray, production: np.ndarray, numerical: bool = True
) -> tuple[float | None, float | None]:
    """Return statistic and p value, or nulls for categorical data."""
    if not numerical:
        return None, None
    result = ks_2samp(reference, production)
    return float(result.statistic), float(result.pvalue)
