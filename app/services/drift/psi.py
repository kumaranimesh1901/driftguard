"""Population Stability Index implementation."""

import numpy as np

from app.config import settings


def _clean(values: np.ndarray, minimum_rows: int = 30) -> np.ndarray:
    """Drop non-finite values and require enough observations."""
    result = np.asarray(values, dtype=float)
    result = result[np.isfinite(result)]
    if result.size < minimum_rows:
        raise ValueError(f"at least {minimum_rows} non-null observations are required")
    return result


def binned_proportions(
    reference: np.ndarray, production: np.ndarray, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return production/reference proportions on open-ended reference bins."""
    ref = _clean(reference)
    prod = _clean(production)
    if edges is None:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, settings.psi_bins + 1)))
    else:
        edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size < 2:
        constant = float(ref[0])
        split = constant + max(abs(constant) * 1e-12, 1e-12)
        bins = np.array([-np.inf, split, np.inf])
    else:
        bins = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
    p = np.histogram(prod, bins=bins)[0]
    q = np.histogram(ref, bins=bins)[0]
    return p / p.sum(), q / q.sum(), bins


def psi(
    reference: np.ndarray, production: np.ndarray, edges: np.ndarray | None = None
) -> float:
    """Compute PSI without dropping values outside reference extrema."""
    p, q, _ = binned_proportions(reference, production, edges)
    p = np.clip(p, 1e-4, None)
    q = np.clip(q, 1e-4, None)
    return float(np.sum((p - q) * np.log(p / q)))


def categorical_psi(reference: np.ndarray, production: np.ndarray) -> float:
    """Compute PSI with an explicit unseen-category bucket."""
    cats = np.unique(reference)
    p = np.array(
        [np.mean(production == c) for c in cats] + [np.mean(~np.isin(production, cats))]
    )
    q = np.array([np.mean(reference == c) for c in cats] + [0.0])
    p = np.clip(p, 1e-4, None)
    q = np.clip(q, 1e-4, None)
    return float(np.sum((p - q) * np.log(p / q)))
