"""ADWIN (adaptive windowing) on explicitly defined model streams.

ADWIN is a *streaming* change detector for a single bounded numeric signal.
It does not detect feature-distribution drift by itself; DriftGuard applies it
to two model-level streams whose meaning is fixed and documented:

``prediction``
    Predicted default probability per record (``[0, 1]``). A change means
    **prediction drift**: the model's output distribution moved. Needs no labels.
``error``
    Absolute error ``|y - p|`` per labelled record (``[0, 1]``). A change means
    the model's error level moved: a **performance signal** that, together with
    the AUC confidence interval, supports confirmed degradation. Needs labels.

Each stream is the baseline *holdout* signal (records the model never trained
on, the no-drift reference) followed by the production batch in stored record
order. Only detections located after the boundary are reported as production
changes; detections inside the reference segment are counted separately as
reference-segment detections (a sanity signal, ideally zero).
"""

from typing import Any

import numpy as np

from app.core.config import settings

try:  # River is the reference implementation (Bifet & Gavaldà, 2007).
    from river.drift import ADWIN

    RIVER_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without river installed
    ADWIN = None
    RIVER_AVAILABLE = False


def adwin_stream(
    reference: np.ndarray,
    production: np.ndarray,
    signal: str,
    delta: float | None = None,
    seed: int = settings.random_seed,
) -> dict[str, Any]:
    """Run ADWIN over ``reference`` then ``production`` and summarise detections.

    The reference segment has no natural time order, so it is shuffled with a
    fixed seed (stationary by construction); production keeps its record order.
    """
    delta = settings.adwin_delta if delta is None else delta
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)
    ref = ref[np.isfinite(ref)]
    prod = prod[np.isfinite(prod)]
    base: dict[str, Any] = {
        "signal": signal,
        "detector": "ADWIN (river)",
        "delta": delta,
        "n_reference": int(ref.size),
        "n_production": int(prod.size),
    }
    if not RIVER_AVAILABLE:
        return {**base, "available": False, "detected": None,
                "reason": "river is not installed"}
    if ref.size == 0 or prod.size == 0:
        return {**base, "available": True, "detected": None,
                "reason": "empty reference or production stream"}
    ref = np.random.default_rng(seed).permutation(ref)
    detector = ADWIN(delta=delta)
    ref_hits: list[int] = []
    prod_hits: list[int] = []
    for i, value in enumerate(np.concatenate([ref, prod])):
        detector.update(float(value))
        if detector.drift_detected:
            (ref_hits if i < ref.size else prod_hits).append(i - ref.size)
    return {
        **base,
        "available": True,
        "detected": bool(prod_hits),
        "n_detections_production": len(prod_hits),
        "n_detections_reference": len(ref_hits),
        # 0-based record position inside the production batch.
        "first_detection_index": prod_hits[0] if prod_hits else None,
        "detection_indices": prod_hits[:50],
        "reference_mean": float(ref.mean()),
        "production_mean": float(prod.mean()),
        "final_window_mean": float(detector.estimation),
        "final_window_width": int(detector.width),
    }
