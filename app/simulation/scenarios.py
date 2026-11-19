"""Named simulator presets."""

from app.simulation.generator import DriftSpec

SCENARIOS = {
    "no_drift": None,
    "income_shift": DriftSpec("income", "mean_shift", 1),
    "loan_amount_shift": DriftSpec("loan_amount", "scale_change", 0.5),
    "credit_score_shift": DriftSpec("credit_score", "mean_shift", -1),
    "employment_mix_change": DriftSpec("employment_type", "proportion_change", 0.2),
    "low_importance_drift": DriftSpec("num_credit_lines", "mean_shift", 1),
    "concept_drift_only": DriftSpec("label_function", "concept_flip", 1),
}
