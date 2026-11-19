"""Jensen-Shannon divergence detector."""

import numpy as np
from scipy.spatial.distance import jensenshannon


def jsd_from_counts(a: np.ndarray, b: np.ndarray) -> float:
    """Return squared base-2 JSD after small smoothing."""
    p = np.asarray(a, dtype=float) + 1e-8
    q = np.asarray(b, dtype=float) + 1e-8
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)
