"""Synthetic banking customer data for demonstrations (always flagged as synthetic).

The 2020 baseline generator and the 2026 production scenarios A-D.
"""

import numpy as np
import pandas as pd

# Scenario catalogue for synthetic production batches (demonstration only).
SCENARIOS = {
    "A": {
        "name": "Stable population",
        "description": "Production distributions remain similar to the 2020 baseline.",
        "expected": "Low drift",
    },
    "B": {
        "name": "Moderate income drift",
        "description": "Monthly income rises ~22%; credit limits (~15%) and loan requests (~12%) rise moderately.",
        "expected": "Moderate drift",
    },
    "C": {
        "name": "Severe credit risk drift",
        "description": "Credit scores fall, outstanding loans and DTI rise, FD balances fall.",
        "expected": "Higher drift impact",
    },
    "D": {
        "name": "Multiple feature drift",
        "description": "Income, credit score, DTI, debt, FD balances and employment mix all shift.",
        "expected": "Significant overall drift",
    },
}


class BankingDataGenerator:
    """Generate realistic 2020 baseline banking loan data and 2026 drifted batches."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate_baseline_2020(self, n: int = 2500) -> pd.DataFrame:
        """Generate 2020 historical reference dataset representing the baseline loan underwriting population."""
        r = self.rng
        customer_ids = [f"CUST-2020-{i+1:05d}" for i in range(n)]
        age = r.integers(21, 65, n)

        # 2020 incomes (in Indian Rupees or equivalent currency units, e.g. monthly ₹25,000 - ₹120,000)
        # Log-normal with mean ~ ₹35,000 monthly
        monthly_income = np.round(r.lognormal(np.log(35_000), 0.40, n), -2)

        employment_types = r.choice(
            ["salaried", "self-employed", "contract", "unemployed"],
            n,
            p=[0.65, 0.22, 0.11, 0.02],
        )

        credit_score = np.clip(np.round(r.normal(710, 60, n)), 300, 850).astype(int)

        # Credit limit is typically 2.5x to 5x monthly income
        credit_limit = np.round(monthly_income * r.uniform(2.5, 4.5, n), -2)

        # Outstanding loans & FD balances
        outstanding_loan_amount = np.round(monthly_income * r.uniform(0.5, 3.5, n), -2)
        fd_balance = np.round(monthly_income * r.uniform(0.5, 6.0, n), -2)

        # Debt-to-income: ratio of monthly EMI to income (~ 0.15 - 0.55 in healthy population)
        debt_to_income = np.clip(np.round(r.beta(2, 5, n), 3), 0.05, 0.90)

        loan_amount_requested = np.round(monthly_income * r.uniform(2.0, 10.0, n), -2)
        loan_tenure = r.choice(
            [12, 24, 36, 48, 60, 84], n, p=[0.1, 0.25, 0.35, 0.15, 0.1, 0.05]
        )
        number_of_existing_loans = r.integers(0, 5, n)

        # Interest rate dependent on credit score
        interest_rate = np.clip(
            np.round(
                0.18 - (credit_score - 300) / 550 * 0.10 + r.normal(0, 0.01, n), 4
            ),
            0.08,
            0.24,
        )

        # Historical Default outcome (calibrated around 12% in 2020)
        # Higher DTI, lower credit score, lower income increase default risk
        risk_score = (
            -0.008 * (credit_score - 700)
            + 2.8 * (debt_to_income - 0.3)
            - 0.00002 * (monthly_income - 35000)
            + 0.00001 * (outstanding_loan_amount - 50000)
            + np.where(employment_types == "unemployed", 1.5, 0)
            + np.where(employment_types == "contract", 0.5, 0)
        )
        default_prob = 1 / (1 + np.exp(-(risk_score - 1.8)))
        loan_default = r.binomial(1, np.clip(default_prob, 0.01, 0.95))

        df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "age": age,
                "monthly_income": monthly_income,
                "employment_type": employment_types,
                "credit_score": credit_score,
                "credit_limit": credit_limit,
                "outstanding_loan_amount": outstanding_loan_amount,
                "fd_balance": fd_balance,
                "debt_to_income": debt_to_income,
                "loan_amount_requested": loan_amount_requested,
                "loan_tenure": loan_tenure,
                "number_of_existing_loans": number_of_existing_loans,
                "interest_rate": interest_rate,
                "loan_default": loan_default,
            }
        )
        return df

    def generate_production_batch(
        self,
        batch_name: str = "January_2026",
        n: int = 1200,
        scenario: str = "A",
        with_labels: bool = False,
        concept_shift: bool = False,
    ) -> pd.DataFrame:
        """
        All generated records are SYNTHETIC and must never be presented as real.

        ``concept_shift=True`` additionally changes the 2026 relationship between
        attributes and default (credit score and FD balance lose most of their
        protective effect; existing-loan count and contract/self-employment
        become dominant drivers). This simulates concept drift so that
        confirmed performance degradation can be demonstrated with labels.

        Generate a 2026 production customer batch under controlled banking drift scenarios:
        - Scenario A: Stable population (Control, minimal drift)
        - Scenario B: Moderate income & inflation drift (salaries up ~22%, credit limits up ~15%)
        - Scenario C: Severe credit risk & debt drift (Credit score down, DTI up, FD down)
        - Scenario D: Multiple feature systemic drift (Macroeconomic crisis, employment mix shift)
        """
        r = self.rng
        df = self.generate_baseline_2020(n)
        prefix = batch_name.replace(" ", "_").replace(".csv", "")
        df["customer_id"] = [f"CUST-2026-{prefix}-{i+1:04d}" for i in range(n)]

        if scenario == "A":
            # Stable population - slight natural noise
            pass

        elif scenario == "B":
            # Scenario B - Moderate income drift (Inflation & salary increments by 2026)
            # Monthly income shifts up ~22%
            df["monthly_income"] = np.round(
                df["monthly_income"] * r.normal(1.22, 0.06, n), -2
            )
            df["credit_limit"] = np.round(
                df["credit_limit"] * r.normal(1.15, 0.08, n), -2
            )
            df["loan_amount_requested"] = np.round(
                df["loan_amount_requested"] * r.normal(1.12, 0.08, n), -2
            )

        elif scenario == "C":
            # Scenario C - Severe credit risk drift
            # Customer debt load escalated, credit scores dropped, FDs liquidated
            df["credit_score"] = np.clip(
                df["credit_score"] - r.integers(40, 90, n), 300, 850
            )
            df["debt_to_income"] = np.clip(
                np.round(df["debt_to_income"] * 1.55, 3), 0.1, 0.98
            )
            df["outstanding_loan_amount"] = np.round(
                df["outstanding_loan_amount"] * 1.75, -2
            )
            df["fd_balance"] = np.round(df["fd_balance"] * 0.45, -2)
            df["number_of_existing_loans"] = np.clip(
                df["number_of_existing_loans"] + r.integers(1, 4, n), 0, 15
            )

        elif scenario == "D":
            # Scenario D - Multi-feature systemic drift
            df["monthly_income"] = np.round(
                df["monthly_income"] * r.normal(1.40, 0.12, n), -2
            )
            df["credit_score"] = np.clip(
                df["credit_score"] - r.integers(50, 110, n), 300, 850
            )
            df["debt_to_income"] = np.clip(
                np.round(df["debt_to_income"] * 1.70, 3), 0.15, 0.99
            )
            df["outstanding_loan_amount"] = np.round(
                df["outstanding_loan_amount"] * 1.90, -2
            )
            df["fd_balance"] = np.round(df["fd_balance"] * 0.35, -2)
            df["number_of_existing_loans"] = np.clip(
                df["number_of_existing_loans"] + r.integers(2, 5, n), 0, 15
            )
            # Significant employment mix change: gig/contract workers increase
            df["employment_type"] = r.choice(
                ["salaried", "self-employed", "contract", "unemployed"],
                n,
                p=[0.38, 0.32, 0.22, 0.08],
            )

        # Recalculate true risk outcomes in 2026
        risk_score = (
            -0.008 * (df["credit_score"] - 700)
            + 3.2 * (df["debt_to_income"] - 0.3)
            - 0.000015 * (df["monthly_income"] - 45000)
            + 0.000012 * (df["outstanding_loan_amount"] - 60000)
            + np.where(df["employment_type"] == "unemployed", 1.8, 0)
            + np.where(df["employment_type"] == "contract", 0.7, 0)
        )
        if concept_shift:
            risk_score = (
                -0.001 * (df["credit_score"] - 700)
                + 0.6 * (df["debt_to_income"] - 0.3)
                + 0.55 * (df["number_of_existing_loans"] - 2)
                + np.where(df["employment_type"] == "self-employed", 1.2, 0)
                + np.where(df["employment_type"] == "contract", 1.0, 0)
                + np.where(df["employment_type"] == "salaried", -0.4, 0)
            )
        default_prob = 1 / (1 + np.exp(-(risk_score - 1.8)))
        df["loan_default"] = r.binomial(1, np.clip(default_prob, 0.02, 0.98))

        if not with_labels:
            # By default, production batches do not have immediate ground truth default labels!
            # They are pending outcome maturation
            df["loan_default"] = np.nan

        return df
