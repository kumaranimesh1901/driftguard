"""Statistical methods: PSI, KS, JSD, normalisation, DIS and the decision engine.

Values are checked against hand calculations and SciPy reference implementations.
"""

import itertools

import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from app.services.drift.jsd import jsd_from_counts
from app.services.drift.ks import ks
from app.services.drift.normalise import aggregate_d, normalise
from app.services.drift.psi import binned_proportions, categorical_psi, psi
from app.services.explain.dis import dis_variants
from app.services.recommendations.decision import decide

RNG = np.random.default_rng(42)
REF = RNG.normal(0, 1, 1000)

# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def test_psi_identical_samples_is_zero() -> None:
    assert psi(REF, REF) < 1e-12


def test_psi_grows_with_shift_and_keeps_out_of_range_mass() -> None:
    p, q, _ = binned_proportions(REF, REF + 3)
    assert p.sum() == pytest.approx(1) and q.sum() == pytest.approx(1)
    assert psi(REF, REF + 3) > psi(REF, REF + 1) > 0.25


def test_psi_matches_hand_formula() -> None:
    p, q, _ = binned_proportions(REF, REF + 0.5)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    assert psi(REF, REF + 0.5) == pytest.approx(np.sum((p - q) * np.log(p / q)))


def test_categorical_psi_binary_shift() -> None:
    ref = np.array(["0"] * 500 + ["1"] * 500)
    prod = np.array(["0"] * 100 + ["1"] * 900)
    expected = (0.1 - 0.5) * np.log(0.1 / 0.5) + (0.9 - 0.5) * np.log(0.9 / 0.5)
    assert categorical_psi(ref, prod) == pytest.approx(expected, rel=1e-3)


def test_constant_reference_change_has_nonzero_psi() -> None:
    assert psi(np.ones(100), np.full(100, 2.0)) > 0


def test_psi_drops_nan_and_rejects_tiny_samples() -> None:
    assert np.isfinite(psi(np.concatenate([REF, [np.nan]]), REF))
    with pytest.raises(ValueError):
        psi(np.ones(10), np.full(10, 2.0))


# ---------------------------------------------------------------------------
# KS and JSD
# ---------------------------------------------------------------------------


def test_ks_matches_scipy_and_is_null_for_categorical() -> None:
    stat, pvalue = ks(REF, REF + 0.5)
    expected = ks_2samp(REF, REF + 0.5)
    assert stat == pytest.approx(expected.statistic)
    assert pvalue == pytest.approx(expected.pvalue)
    assert ks(REF, REF, numerical=False) == (None, None)


def test_ks_ignores_missing_values() -> None:
    prod = RNG.normal(size=500)
    prod[:10] = np.nan
    stat, pvalue = ks(REF, prod)
    assert np.isfinite(stat) and np.isfinite(pvalue)
    assert (stat, pvalue) == ks(REF, prod[10:])


def test_jsd_is_symmetric_zero_on_identity_and_matches_scipy() -> None:
    a, b = np.array([10, 20, 30, 40]), np.array([40, 30, 20, 10])
    assert jsd_from_counts(a, b) == pytest.approx(jsd_from_counts(b, a))
    assert jsd_from_counts(a, a) < 1e-12
    pa, pb = (a + 1e-8) / (a + 1e-8).sum(), (b + 1e-8) / (b + 1e-8).sum()
    assert jsd_from_counts(a, b) == pytest.approx(jensenshannon(pa, pb, base=2) ** 2)


# ---------------------------------------------------------------------------
# Normalisation and DIS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"), [(0.02, 0), (0.05, 0), (0.275, 0.5), (0.5, 1), (0.8, 1)]
)
def test_normalise_maps_floor_to_cap_onto_unit_interval(value, expected) -> None:
    assert normalise(value, 0.05, 0.5) == pytest.approx(expected)


def test_normalise_is_monotonic_and_aggregates() -> None:
    xs = np.linspace(0, 1, 50)
    ys = [normalise(x, 0.05, 0.5) for x in xs]
    assert all(a <= b for a, b in itertools.pairwise(ys))
    assert aggregate_d([0.2, 0.6], "max") == 0.6
    assert aggregate_d([0.2, 0.6], "mean") == pytest.approx(0.4)
    assert aggregate_d([]) == 0.0


def test_dis_and_variants_match_hand_calculation() -> None:
    d = {"income": 0.4, "credit_score": 0.8, "age": 0.1}
    fi = {"income": 0.6, "credit_score": 0.5, "age": 0.2}
    s = {"income": 0.5, "credit_score": 0.9, "age": 0.3}
    out = dis_variants(d, fi, s)
    assert out["income"]["dis"] == pytest.approx(0.4 * 0.6 * 0.5)
    assert out["income"]["d_only"] == pytest.approx(0.4)
    assert out["income"]["d_fi"] == pytest.approx(0.24)
    assert out["income"]["weighted_sum"] == pytest.approx(0.475)
    assert all(0 <= v <= 1 for row in out.values() for v in row.values())


def test_dis_rejects_mismatched_or_out_of_range_inputs() -> None:
    with pytest.raises(ValueError):
        dis_variants({"a": 0.5}, {"b": 0.5}, {"a": 0.5})
    with pytest.raises(ValueError):
        dis_variants({"a": 1.5}, {"a": 0.5}, {"a": 0.5})


def test_rank_variant_uses_average_ranks_for_ties() -> None:
    values = {"a": 0.5, "b": 0.5, "c": 0.5}
    assert {
        row["rank_based"] for row in dis_variants(values, values, values).values()
    } == {0.5}


# ---------------------------------------------------------------------------
# Decision engine (SPEC section 13 table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "perf", "alert", "text"),
    [
        ("low", "unknown", None, "Continue monitoring"),
        ("low", "stable", None, "Continue monitoring"),
        (
            "moderate",
            "unknown",
            "drift-warning",
            "Investigate and monitor closely; labels needed to confirm impact",
        ),
        (
            "high",
            "unknown",
            "drift-warning",
            "Investigate affected features and data quality; labels needed to confirm impact",
        ),
        ("moderate", "stable", "drift-warning", "Investigate and monitor closely"),
        (
            "high",
            "stable",
            "drift-warning",
            "Investigate affected features and data quality",
        ),
        ("moderate", "degraded", "confirmed-degradation", "Consider retraining"),
        ("high", "degraded", "confirmed-degradation", "Consider retraining"),
        (
            "low",
            "severe",
            "confirmed-degradation",
            "Immediate model review and controlled retraining",
        ),
        (
            "high",
            "severe",
            "confirmed-degradation",
            "Immediate model review and controlled retraining",
        ),
    ],
)
def test_decision_table(level, perf, alert, text) -> None:
    alert_type, _, recommendation = decide(level, perf)
    assert alert_type == alert and recommendation == text


def test_degradation_without_drift_points_to_concept_drift() -> None:
    alert_type, _, recommendation = decide("low", "degraded")
    assert alert_type == "confirmed-degradation" and "concept drift" in recommendation
