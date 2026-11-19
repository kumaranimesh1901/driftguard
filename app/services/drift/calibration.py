"""Bootstrap no-drift noise floors for PSI, KS and JSD (SPEC section 10.1).

For a feature and a production batch size ``n``, pairs of samples are drawn
from the reference population only, so any measured difference between them
is pure sampling noise. The ``NULL_QUANTILE`` of the resulting statistic is the
noise floor used by :func:`app.services.drift.normalise.normalise`.
"""

import logging
from functools import lru_cache

import numpy as np
from scipy.stats import ks_2samp

from app.core.config import settings
from app.services.drift.jsd import jsd_from_counts
from app.services.drift.psi import binned_proportions, categorical_psi

logger = logging.getLogger(__name__)


def batch_size_bucket(n: int) -> int:
    """Round a batch size to two significant figures so floors can be cached."""
    if n < 100:
        return max(int(n), 1)
    magnitude = 10 ** (len(str(int(n))) - 2)
    return int(round(n / magnitude) * magnitude)


def _draw_pair(
    ref: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two reference-like samples, disjoint halves when the sample allows."""
    if 2 * n <= ref.size:
        idx = rng.permutation(ref.size)
        return ref[idx[:n]], ref[idx[n : 2 * n]]
    half = ref.size // 2
    idx = rng.permutation(ref.size)
    a, b = ref[idx[:half]], ref[idx[half:]]
    return rng.choice(a, n, replace=True), rng.choice(b, n, replace=True)


def numerical_noise_floors(
    reference: np.ndarray,
    n_batch: int,
    b: int = settings.null_bootstrap_b,
    quantile: float = settings.null_quantile,
    seed: int = settings.random_seed,
) -> dict[str, float]:
    """Return the no-drift ``quantile`` of PSI, KS and JSD for batch size ``n_batch``."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    rng = np.random.default_rng(seed)
    n = max(int(n_batch), 30)
    psis, kss, jsds = [], [], []
    for _ in range(b):
        x, y = _draw_pair(ref, n, rng)
        p, q, _ = binned_proportions(x, y)
        pc, qc = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
        psis.append(float(np.sum((pc - qc) * np.log(pc / qc))))
        kss.append(float(ks_2samp(x, y).statistic))
        jsds.append(jsd_from_counts(q, p))
    return {
        "psi": float(np.quantile(psis, quantile)),
        "ks": float(np.quantile(kss, quantile)),
        "jsd": float(np.quantile(jsds, quantile)),
    }


def categorical_noise_floors(
    reference: np.ndarray,
    n_batch: int,
    b: int = settings.null_bootstrap_b,
    quantile: float = settings.null_quantile,
    seed: int = settings.random_seed,
) -> dict[str, float]:
    """Return the no-drift ``quantile`` of categorical PSI and JSD."""
    ref = np.asarray(reference, dtype=object)
    rng = np.random.default_rng(seed)
    n = max(int(n_batch), 30)
    cats = np.unique(ref)
    psis, jsds = [], []
    for _ in range(b):
        x, y = _draw_pair(ref, n, rng)
        psis.append(categorical_psi(x, y))
        cx = np.array([np.sum(x == c) for c in cats])
        cy = np.array([np.sum(y == c) for c in cats])
        jsds.append(jsd_from_counts(cx, cy))
    return {
        "psi": float(np.quantile(psis, quantile)),
        "ks": None,
        "jsd": float(np.quantile(jsds, quantile)),
    }


@lru_cache(maxsize=512)
def _cached(
    key: str, kind: str, n_bucket: int, values: tuple
) -> tuple[tuple[str, float | None], ...]:
    arr = np.array(values, dtype=float if kind == "numerical" else object)
    fn = numerical_noise_floors if kind == "numerical" else categorical_noise_floors
    return tuple(fn(arr, n_bucket).items())


def noise_floors(
    cache_key: str, reference: np.ndarray, n_batch: int, kind: str
) -> dict[str, float | None]:
    """Cached noise floors keyed by ``(cache_key, kind, batch-size bucket)``.

    ``cache_key`` must identify the reference sample (e.g. ``"ds12:income"``).
    """
    bucket = batch_size_bucket(n_batch)
    values = tuple(reference.tolist())
    return dict(_cached(cache_key, kind, bucket, values))
