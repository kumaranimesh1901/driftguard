"""Regression coverage for confirmed first-pass defects."""

import numpy as np
import pytest

from app.services.drift.psi import binned_proportions, psi
from app.services.explain.dis import dis_variants
from app.simulation.generator import DriftSpec, LoanDataGenerator


def test_psi_retains_out_of_range_production_mass() -> None:
    """D1: tail-shift mass is placed in the open-ended bins."""
    rng = np.random.default_rng(7)
    reference = rng.normal(size=1_000)
    p, q, _ = binned_proportions(reference, reference + 3)
    assert p.sum() == pytest.approx(1)
    assert q.sum() == pytest.approx(1)
    assert psi(reference, reference + 3) > psi(reference, reference + 1)


def test_constant_reference_change_has_nonzero_psi() -> None:
    """D1: constant references receive a two-sided fallback bin."""
    assert psi(np.ones(100), np.full(100, 2.0)) > 0


def test_generator_default_rate_is_calibrated() -> None:
    """D2: baseline synthetic defaults match the documented realistic range."""
    rate = LoanDataGenerator(3).generate(20_000).default.mean()
    assert 0.13 <= rate <= 0.22


@pytest.mark.parametrize("magnitude", [0, 0.2, 0.6, 1.0])
def test_employment_mix_accepts_valid_magnitudes(magnitude: float) -> None:
    """D4: interpolated category mixes remain valid distributions."""
    generator = LoanDataGenerator(9)
    data = generator.generate(
        2_000, DriftSpec("employment_type", "proportion_change", magnitude)
    )
    assert len(data) == 2_000


def test_invalid_drift_spec_is_rejected() -> None:
    """D5: unsupported interventions cannot make false ground-truth claims."""
    with pytest.raises(ValueError):
        DriftSpec("income", "scale_change", 1)


def test_rank_variant_uses_average_ranks_for_ties() -> None:
    """D11: tied scores are independent of feature insertion order."""
    values = {"a": 0.5, "b": 0.5, "c": 0.5}
    result = dis_variants(values, values, values)
    assert {row["rank_based"] for row in result.values()} == {0.5}
