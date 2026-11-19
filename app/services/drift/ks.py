"""Kolmogorov-Smirnov detector."""

import numpy as np
from scipy.stats import ks_2samp

from app.services.drift.psi import _clean


def ks(
    reference: np.ndarray, production: np.ndarray, numerical: bool = True
) -> tuple[float | None, float | None]:
    """Return statistic and p value, or nulls for categorical data.

    Non-finite values are dropped (as in PSI) so a few missing values cannot
    turn the result into NaN.
    """
    if not numerical:
        return None, None
    result = ks_2samp(_clean(reference), _clean(production))
    return float(result.statistic), float(result.pvalue)
