"""Deterministic synthetic loan data with validated isolated interventions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftSpec:
    feature: str
    kind: str
    magnitude: float
    pattern: str = "sudden"
    onset_batch: int = 0
    duration_batches: int = 1
    period_batches: int = 2

    def __post_init__(self) -> None:
        allowed = {
            ("income", "mean_shift"),
            ("credit_score", "mean_shift"),
            ("loan_amount", "scale_change"),
            ("num_credit_lines", "mean_shift"),
            ("employment_type", "proportion_change"),
            ("home_ownership", "proportion_change"),
            ("label_function", "concept_flip"),
        }
        if (self.feature, self.kind) not in allowed:
            raise ValueError("unsupported feature/kind drift combination")
        if self.pattern not in {"sudden", "gradual", "recurring", "none"}:
            raise ValueError("unsupported drift pattern")
        if self.duration_batches < 1 or self.period_batches < 1:
            raise ValueError("duration and period must be positive")
        if self.kind == "proportion_change" and self.magnitude < 0:
            raise ValueError("proportion change magnitude must be non-negative")


class LoanDataGenerator:
    """Generate loans with a default-risk intercept calibrated to 17 percent."""

    numerical = (
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "debt_to_income",
        "employment_years",
        "num_credit_lines",
        "interest_rate",
    )
    categorical = ("employment_type", "home_ownership", "loan_purpose")

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.coefficients = {
            "credit_score": -0.012,
            "debt_to_income": 3.0,
            "loan_to_income": 1.2,
            "income": -0.000005,
            "employment_years": -0.03,
        }
        calibration = self._base(50_000, np.random.default_rng(seed + 1))
        terms = self._terms(calibration)
        low, high = -20.0, 10.0
        for _ in range(60):
            mid = (low + high) / 2
            if self._sigmoid(mid + terms).mean() > 0.17:
                high = mid
            else:
                low = mid
        self.intercept = (low + high) / 2

    @staticmethod
    def _sigmoid(v: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(v, -700, 700)))

    def _base(self, n: int, r: np.random.Generator) -> pd.DataFrame:
        income = r.lognormal(np.log(45_000), 0.45, n)
        credit = np.clip(r.normal(680, 70, n), 300, 850)
        return pd.DataFrame(
            {
                "age": r.integers(21, 76, n),
                "income": income,
                "loan_amount": income * r.lognormal(-0.2, 0.45, n),
                "credit_score": credit,
                "debt_to_income": r.beta(2, 5, n),
                "employment_years": r.integers(0, 41, n),
                "num_credit_lines": r.integers(0, 16, n),
                "interest_rate": np.clip(
                    0.26 - credit / 5000 + r.normal(0, 0.015, n), 0.03, 0.36
                ),
                "employment_type": r.choice(
                    ["salaried", "self-employed", "contract", "unemployed"],
                    n,
                    p=[0.62, 0.20, 0.14, 0.04],
                ),
                "home_ownership": r.choice(
                    ["rent", "own", "mortgage"], n, p=[0.42, 0.22, 0.36]
                ),
                "loan_purpose": r.choice(
                    ["education", "home", "medical", "business", "personal"], n
                ),
            }
        )

    def _terms(self, data: pd.DataFrame) -> np.ndarray:
        c = self.coefficients
        return (
            c["credit_score"] * (data.credit_score - 680)
            + c["debt_to_income"] * data.debt_to_income
            + c["loan_to_income"] * data.loan_amount / data.income
            + c["income"] * (data.income - 45_000)
            + c["employment_years"] * data.employment_years
            + np.where(data.employment_type == "unemployed", 1.4, 0)
            + np.where(data.employment_type == "self-employed", 0.45, 0)
        )

    def _amount(self, d: DriftSpec | None, batch: int) -> float:
        if not d or batch < d.onset_batch or d.pattern == "none":
            return 0.0
        if d.pattern == "gradual":
            return d.magnitude * min(
                1, (batch - d.onset_batch + 1) / d.duration_batches
            )
        if d.pattern == "recurring":
            return (
                d.magnitude
                if ((batch - d.onset_batch) // d.period_batches) % 2 == 0
                else 0.0
            )
        return d.magnitude

    def generate(
        self, n: int, drift: DriftSpec | None = None, batch_index: int = 0
    ) -> pd.DataFrame:
        """Generate records; dependent fields are fixed before a single feature is drifted."""
        data = self._base(n, self.rng)
        amount = self._amount(drift, batch_index)
        if drift and amount:
            if drift.feature == "income":
                data["income"] *= np.exp(amount * 0.45)
            elif drift.feature == "credit_score":
                data["credit_score"] = np.clip(
                    data.credit_score + amount * 70, 300, 850
                )
            elif drift.feature == "loan_amount":
                data["loan_amount"] *= 1 + amount
            elif drift.feature == "num_credit_lines":
                data["num_credit_lines"] = np.clip(
                    data.num_credit_lines + round(amount * 4), 0, 15
                )
            elif drift.feature in {"employment_type", "home_ownership"}:
                values, base, target = (
                    (
                        ["salaried", "self-employed", "contract", "unemployed"],
                        [0.62, 0.20, 0.14, 0.04],
                        [0.30, 0.40, 0.14, 0.16],
                    )
                    if drift.feature == "employment_type"
                    else (
                        ["rent", "own", "mortgage"],
                        [0.42, 0.22, 0.36],
                        [0.65, 0.10, 0.25],
                    )
                )
                probabilities = (1 - min(amount, 1)) * np.array(base) + min(
                    amount, 1
                ) * np.array(target)
                data[drift.feature] = self.rng.choice(
                    values, n, p=probabilities / probabilities.sum()
                )
        terms = self._terms(data)
        if drift and drift.feature == "label_function":
            terms += amount * (data.debt_to_income * 3 - 1)
        data["default"] = self.rng.binomial(1, self._sigmoid(self.intercept + terms))
        data.attrs["ground_truth"] = {
            "drifting_features": (
                []
                if not drift or not amount or drift.feature == "label_function"
                else [drift.feature]
            ),
            "concept_drift": bool(
                drift and drift.feature == "label_function" and amount
            ),
            "magnitude": amount,
            "pattern": drift.pattern if drift else "none",
        }
        return data
