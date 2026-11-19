"""Banking dataset schema, validation, cleaning and synthetic data generation.

Required attributes of a banking credit-risk record:

- ``customer_id`` (identifier, never used as a drift feature)
- ``age``, ``monthly_income``, ``credit_score``, ``credit_limit``,
  ``outstanding_loan_amount``, ``fd_balance``, ``debt_to_income``,
  ``loan_amount_requested``, ``loan_tenure``, ``number_of_existing_loans``,
  ``interest_rate`` (numerical)
- ``employment_type`` (categorical)
- ``loan_default`` (optional 0/1 outcome label; blank while outcomes are pending)
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMN = "customer_id"
LABEL_COLUMN = "loan_default"

NUMERICAL_FEATURES = [
    "age",
    "monthly_income",
    "credit_score",
    "credit_limit",
    "outstanding_loan_amount",
    "fd_balance",
    "debt_to_income",
    "loan_amount_requested",
    "loan_tenure",
    "number_of_existing_loans",
    "interest_rate",
]

CATEGORICAL_FEATURES = ["employment_type"]

MONITORED_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

BANKING_REQUIRED_COLUMNS = [ID_COLUMN, *MONITORED_FEATURES]

BANKING_OPTIONAL_COLUMNS = [LABEL_COLUMN]

FEATURE_LABELS = {
    "age": "Age",
    "monthly_income": "Monthly Income",
    "employment_type": "Employment Type",
    "credit_score": "Credit Score",
    "credit_limit": "Credit Limit",
    "outstanding_loan_amount": "Outstanding Loan Amount",
    "fd_balance": "FD Balance",
    "debt_to_income": "Debt-to-Income",
    "loan_amount_requested": "Loan Amount Requested",
    "loan_tenure": "Loan Tenure (months)",
    "number_of_existing_loans": "Existing Loans",
    "interest_rate": "Interest Rate",
}

KNOWN_EMPLOYMENT_TYPES = {"salaried", "self-employed", "contract", "unemployed"}

# Plausibility ranges; values outside are reported as warnings, never silently fixed.
PLAUSIBLE_RANGES = {
    "age": (18, 100),
    "monthly_income": (0, None),
    "credit_score": (300, 900),
    "credit_limit": (0, None),
    "outstanding_loan_amount": (0, None),
    "fd_balance": (0, None),
    "debt_to_income": (0, 5),
    "loan_amount_requested": (0, None),
    "loan_tenure": (1, 600),
    "number_of_existing_loans": (0, 100),
    "interest_rate": (0, 1),
}

MIN_ROWS = 30
# Above this share of unparseable values a numerical column is rejected.
MAX_UNPARSEABLE_SHARE = 0.05

BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")


@dataclass
class ValidationResult:
    """Outcome of validating an uploaded banking CSV."""

    is_valid: bool
    missing_columns: list[str]
    row_count: int
    column_count: int
    errors: list[str]
    warnings: list[str]
    preview: list[dict[str, Any]]
    has_labels: bool = False
    labelled_rows: int = 0
    synthetic_columns: list[str] = field(default_factory=list)
    column_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and snake-case column headers so real exports map cleanly."""
    out = df.copy()
    out.columns = [
        re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")
        for c in out.columns
    ]
    return out


def json_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert rows to JSON-safe dicts (NaN/inf become ``None``)."""
    part = df if limit is None else df.head(limit)
    part = part.replace([np.inf, -np.inf], np.nan).astype(object)
    part = part.where(pd.notna(part), None)
    return part.to_dict(orient="records")


def prepare_banking_frame(
    raw: pd.DataFrame,
    fill_missing_with_synthetic: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, ValidationResult]:
    """Validate, clean and type-coerce a banking dataset.

    Returns the cleaned frame (required + optional columns only) and the
    validation report. The frame must not be used when ``is_valid`` is false.
    """
    errors: list[str] = []
    warnings: list[str] = []
    df = normalise_headers(raw)

    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        errors.append(f"Duplicate column names after normalisation: {', '.join(dupes)}")
        df = df.loc[:, ~df.columns.duplicated()]

    if len(df) == 0:
        errors.append("Dataset is empty (no customer records).")

    synthetic_columns: list[str] = []
    missing = [c for c in BANKING_REQUIRED_COLUMNS if c not in df.columns]
    if missing and fill_missing_with_synthetic and len(df) > 0:
        df, synthetic_columns = augment_missing_columns(df, seed=seed)
        warnings.append(
            "SYNTHETIC DATA: the following columns were absent and were generated "
            f"synthetically for demonstration only: {', '.join(synthetic_columns)}."
        )
        missing = []
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return df, _result(df, errors, warnings, missing, synthetic_columns)

    extra = [
        c
        for c in df.columns
        if c not in BANKING_REQUIRED_COLUMNS + BANKING_OPTIONAL_COLUMNS
    ]
    if extra:
        warnings.append(f"Ignored unrecognised columns: {', '.join(extra)}")

    n = len(df)
    if 0 < n < MIN_ROWS:
        errors.append(
            f"Dataset has {n} records; at least {MIN_ROWS} are required for "
            "statistically meaningful drift detection."
        )

    clean = pd.DataFrame(index=df.index)
    clean[ID_COLUMN] = df[ID_COLUMN].astype("string").str.strip()
    if clean[ID_COLUMN].isna().any():
        warnings.append(
            f"{int(clean[ID_COLUMN].isna().sum())} records have no customer_id."
        )
    dup_ids = int(clean[ID_COLUMN].dropna().duplicated().sum())
    if dup_ids:
        warnings.append(f"{dup_ids} duplicate customer_id values found.")

    for col in NUMERICAL_FEATURES:
        original = df[col]
        if original.dtype == object or pd.api.types.is_string_dtype(original):
            original = original.astype("string").str.strip().replace("", pd.NA)
        values = pd.to_numeric(original, errors="coerce")
        unparseable = int((values.isna() & original.notna()).sum())
        if n and unparseable:
            share = unparseable / n
            msg = f"Column '{col}' has {unparseable} ({share:.1%}) non-numeric values"
            if share > MAX_UNPARSEABLE_SHARE:
                errors.append(msg + "; the column type is incompatible.")
            else:
                warnings.append(msg + "; they are treated as missing.")
        infinite = int(np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).sum())
        if infinite:
            warnings.append(
                f"Column '{col}' has {infinite} infinite values; treated as missing."
            )
            values = values.replace([np.inf, -np.inf], np.nan)
        if n and values.notna().sum() == 0:
            errors.append(f"Column '{col}' contains no usable numeric values.")
        lo, hi = PLAUSIBLE_RANGES[col]
        out_of_range = 0
        if lo is not None:
            out_of_range += int((values < lo).sum())
        if hi is not None:
            out_of_range += int((values > hi).sum())
        if out_of_range:
            warnings.append(
                f"Column '{col}' has {out_of_range} values outside the plausible "
                f"range [{lo}, {hi if hi is not None else '∞'}]."
            )
        clean[col] = values.astype(float)

    emp = df["employment_type"].astype("string").str.strip().str.lower()
    emp = emp.replace("", pd.NA)
    clean["employment_type"] = emp
    if n and emp.notna().sum() == 0:
        errors.append("Column 'employment_type' contains no values.")
    unknown = sorted(set(emp.dropna().unique()) - KNOWN_EMPLOYMENT_TYPES)
    if unknown:
        warnings.append(
            "Unrecognised employment_type categories (monitored as new categories): "
            + ", ".join(unknown[:10])
        )

    labelled = 0
    if LABEL_COLUMN in df.columns:
        label_raw = df[LABEL_COLUMN]
        if label_raw.dtype == object or pd.api.types.is_string_dtype(label_raw):
            label_raw = label_raw.astype("string").str.strip().replace("", pd.NA)
        labels = pd.to_numeric(label_raw, errors="coerce")
        bad = int((labels.notna() & ~labels.isin([0, 1])).sum()) + int(
            (labels.isna() & label_raw.notna()).sum()
        )
        if bad:
            errors.append(
                f"Column 'loan_default' must contain only 0, 1 or blank; {bad} invalid values."
            )
        clean[LABEL_COLUMN] = labels.where(labels.isin([0, 1]))
        labelled = int(clean[LABEL_COLUMN].notna().sum())
        if 0 < labelled < n:
            warnings.append(
                f"{n - labelled} records have no loan_default outcome (pending)."
            )
    else:
        clean[LABEL_COLUMN] = np.nan

    for col in MONITORED_FEATURES:
        missing_n = int(clean[col].isna().sum())
        if n and missing_n and f"Column '{col}' contains no" not in " ".join(errors):
            warnings.append(
                f"Column '{col}' has {missing_n} ({missing_n / n:.1%}) missing values; "
                "they are excluded per feature from drift statistics and imputed for "
                "model scoring."
            )

    return clean, _result(
        clean, errors, warnings, [], synthetic_columns, labelled_rows=labelled
    )


def _result(
    df: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
    missing: list[str],
    synthetic_columns: list[str],
    labelled_rows: int = 0,
) -> ValidationResult:
    summary = [
        {
            "column": str(c),
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "synthetic": c in synthetic_columns,
        }
        for c in df.columns
    ]
    return ValidationResult(
        is_valid=not errors,
        missing_columns=missing,
        row_count=len(df),
        column_count=len(df.columns),
        errors=errors,
        warnings=warnings,
        preview=json_records(df, 10),
        has_labels=labelled_rows > 0,
        labelled_rows=labelled_rows,
        synthetic_columns=synthetic_columns,
        column_summary=summary,
    )


def validate_banking_dataset(df: pd.DataFrame) -> ValidationResult:
    """Validate a banking dataframe (no synthetic augmentation)."""
    return prepare_banking_frame(df)[1]


def augment_missing_columns(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, list[str]]:
    """Fill absent required attributes with clearly flagged synthetic values.

    Real columns and real ``loan_default`` labels are preserved untouched;
    labels are never synthesised here. Generated columns are conditioned on
    real income/credit score when those columns exist.
    """
    r = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    added: list[str] = []

    def num(col: str) -> np.ndarray | None:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        return None

    def add(col: str, values: Any) -> None:
        out[col] = values
        added.append(col)

    if ID_COLUMN not in out.columns:
        add(ID_COLUMN, [f"SYN-ID-{i + 1:06d}" for i in range(n)])
    if "age" not in out.columns:
        add("age", r.integers(21, 65, n))
    if "monthly_income" not in out.columns:
        add("monthly_income", np.round(r.lognormal(np.log(35_000), 0.40, n), -2))
    income = num("monthly_income")
    income = np.where(np.isfinite(income), income, np.nanmedian(income))
    if "employment_type" not in out.columns:
        add(
            "employment_type",
            r.choice(
                ["salaried", "self-employed", "contract", "unemployed"],
                n,
                p=[0.65, 0.22, 0.11, 0.02],
            ),
        )
    if "credit_score" not in out.columns:
        add("credit_score", np.clip(np.round(r.normal(710, 60, n)), 300, 850))
    score = num("credit_score")
    score = np.where(np.isfinite(score), score, np.nanmedian(score))
    generators = {
        "credit_limit": lambda: np.round(income * r.uniform(2.5, 4.5, n), -2),
        "outstanding_loan_amount": lambda: np.round(
            income * r.uniform(0.5, 3.5, n), -2
        ),
        "fd_balance": lambda: np.round(income * r.uniform(0.5, 6.0, n), -2),
        "debt_to_income": lambda: np.clip(np.round(r.beta(2, 5, n), 3), 0.05, 0.9),
        "loan_amount_requested": lambda: np.round(income * r.uniform(2.0, 10.0, n), -2),
        "loan_tenure": lambda: r.choice(
            [12, 24, 36, 48, 60, 84], n, p=[0.1, 0.25, 0.35, 0.15, 0.1, 0.05]
        ),
        "number_of_existing_loans": lambda: r.integers(0, 5, n),
        "interest_rate": lambda: np.clip(
            np.round(0.18 - (score - 300) / 550 * 0.10 + r.normal(0, 0.01, n), 4),
            0.08,
            0.24,
        ),
    }
    for col, make in generators.items():
        if col not in out.columns:
            add(col, make())
    return out, added
