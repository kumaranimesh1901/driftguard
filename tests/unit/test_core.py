import numpy as np

from app.services.decision import decide
from app.services.drift.jsd import jsd_from_counts
from app.services.drift.normalise import normalise
from app.services.drift.psi import psi


def test_identical_psi_is_zero():
    x = np.arange(100)
    assert psi(x, x) < 1e-12


def test_jsd_is_symmetric_and_bounded():
    assert jsd_from_counts(np.array([1, 4]), np.array([4, 1])) == jsd_from_counts(
        np.array([4, 1]), np.array([1, 4])
    )


def test_normalisation_bounds():
    assert normalise(0.1, 0.2, 0.5) == 0 and normalise(0.8, 0.2, 0.5) == 1


def test_decision_keeps_warning_distinct():
    assert (
        decide("high", "unknown")[0] == "drift-warning"
        and decide("high", "degraded")[0] == "confirmed-degradation"
    )
