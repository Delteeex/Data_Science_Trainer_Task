"""Stateless normalization and validation for the raw FinTrust loan datasets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "defaulted"
SCENARIO_COLUMN = "scenario"

FEATURE_COLUMNS = (
    "applicant_id",
    "application_date",
    "age",
    "gender",
    "marital_status",
    "dependents",
    "education",
    "employment_type",
    "employer_category",
    "years_at_current_job",
    "annual_income",
    "city_tier",
    "residence_type",
    "years_at_residence",
    "loan_purpose",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "num_credit_lines",
    "existing_monthly_emi",
    "late_payments_last_2y",
    "credit_utilization",
    "hard_inquiries_6m",
    "bank_relationship_months",
    "has_cosigner",
    "has_collateral",
    "application_channel",
    "branch_code",
    "debt_to_income",
    "recovery_calls_made",
    "country",
    "loan_amount_lakhs",
    "record_created_by",
)

EXCLUDED_FEATURE_REASONS = {
    "applicant_id": "Pure identifier; retained for row linkage, never used as a predictor.",
    "defaulted": "Target label; never part of the feature matrix.",
    "recovery_calls_made": (
        "Post-application recovery activity; direct temporal target leakage."
    ),
    "record_created_by": (
        "Operational record metadata, not applicant information; proxy/leakage risk."
    ),
    "country": "Constant (India) in all supplied datasets.",
    "loan_amount_lakhs": (
        "Redundant alternate-unit copy of loan_amount; inconsistent in scenario rows."
    ),
    "interest_rate": (
        "Offered rate may be assigned after the application-time decision; timing is unconfirmed."
    ),
    "scenario": "Hand-written scenario narrative, not an application-time feature.",
}

_TARGET_MAP = {
    "1": 1,
    "1.0": 1,
    "true": 1,
    "yes": 1,
    "y": 1,
    "default": 1,
    "0": 0,
    "0.0": 0,
    "false": 0,
    "no": 0,
    "n": 0,
    "non-default": 0,
}

_YES_NO_MAP = {
    "yes": "Yes",
    "y": "Yes",
    "true": "Yes",
    "1": "Yes",
    "no": "No",
    "n": "No",
    "false": "No",
    "0": "No",
}

_CATEGORY_ALIASES = {
    "gender": {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female",
    },
    "marital_status": {
        "m": "Married",
        "married": "Married",
        "s": "Single",
        "single": "Single",
        "unmarried": "Single",
        "divorced": "Divorced",
        "separated": "Separated",
        "widow": "Widowed",
        "widowed": "Widowed",
    },
    "education": {
        "grad": "Graduate",
        "graduate": "Graduate",
        "post graduate": "Post Graduate",
        "post-graduate": "Post Graduate",
        "pg": "Post Graduate",
        "masters": "Post Graduate",
        "master's": "Post Graduate",
        "10": "10th",
        "matric": "10th",
        "ssc": "10th",
        "12": "12th",
        "hsc": "12th",
        "intermediate": "12th",
    },
    "employment_type": {
        "salaried": "Salaried",
        "salaried employee": "Salaried",
        "service": "Salaried",
        "self-employed": "Self-Employed",
        "self employed": "Self-Employed",
        "se": "Self-Employed",
        "business": "Business Owner",
        "business owner": "Business Owner",
        "freelancer": "Freelancer",
        "freelance": "Freelancer",
        "gig worker": "Freelancer",
        "retired": "Retired",
        "pensioner": "Retired",
    },
    "city_tier": {
        "1": "Tier 1",
        "t1": "Tier 1",
        "tier 1": "Tier 1",
        "metro": "Tier 1",
        "2": "Tier 2",
        "t2": "Tier 2",
        "tier2": "Tier 2",
        "tier 2": "Tier 2",
        "3": "Tier 3",
        "t3": "Tier 3",
        "tier3": "Tier 3",
        "tier 3": "Tier 3",
        "rural/semi-urban": "Tier 3",
    },
    "residence_type": {
        "rent": "Rented",
        "rented": "Rented",
        "own": "Owned",
        "owned": "Owned",
        "self-owned": "Owned",
        "parental": "Parental",
        "parents": "Parental",
        "family owned": "Parental",
        "company": "Company Provided",
        "company lease": "Company Provided",
        "company provided": "Company Provided",
    },
    "loan_purpose": {
        "debt consolidaton": "Debt Consolidation",
        "debt consolidation": "Debt Consolidation",
        "edu loan": "Education",
        "vehicle": "Vehicle Purchase",
    },
    "application_channel": {
        "web": "Online",
        "website": "Online",
        "online": "Online",
        "branch": "Branch",
        "walk-in": "Branch",
        "dsa": "DSA Agent",
        "agent": "DSA Agent",
        "dsa agent": "DSA Agent",
        "partner": "Partner App",
        "partner app": "Partner App",
        "fintech partner": "Partner App",
    },
}

_CATEGORICAL_COLUMNS = (
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "employer_category",
    "city_tier",
    "residence_type",
    "loan_purpose",
    "has_cosigner",
    "has_collateral",
    "application_channel",
    "branch_code",
    "country",
)
_RATIO_FRACTION_UPPER_BOUND = 1.5

_NUMERIC_COLUMNS = (
    "age",
    "dependents",
    "years_at_current_job",
    "annual_income",
    "years_at_residence",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "credit_score_no_history",
    "num_credit_lines",
    "existing_monthly_emi",
    "late_payments_last_2y",
    "credit_utilization",
    "hard_inquiries_6m",
    "bank_relationship_months",
    "debt_to_income",
    "application_year",
    "application_month",
    "loan_amount_lakhs",
    "recovery_calls_made",
)


def _default_data_dir() -> Path:
    """Resolve the supplied data directory from this module's project location."""
    return Path(__file__).resolve().parents[3] / "DS_Trainer_Task2_Data"


def load_datasets(
    data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load the original train, test, and scenario CSVs without modifying them."""
    directory = Path(data_dir) if data_dir is not None else _default_data_dir()
    paths = {
        "train": directory / "loan_train.csv",
        "test": directory / "loan_test.csv",
        "scenario": directory / "loan_scenario_cases.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required loan dataset(s) not found: " + ", ".join(missing))
    return {
        name: pd.read_csv(path, low_memory=False)
        for name, path in paths.items()
    }


def _clean_text(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text == "-":
        return np.nan
    return text


def _extract_number(value: Any) -> float:
    """Parse a numeric token from dirty text; units are handled by each field."""
    text = _clean_text(value)
    if not isinstance(text, str):
        return np.nan
    text = text.replace(",", "").replace("₹", "").replace("Rs.", "").replace("INR", "")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text, flags=re.IGNORECASE)
    return float(match.group()) if match else np.nan


def _parse_annual_income(value: Any) -> float:
    text = _clean_text(value)
    number = _extract_number(text)
    if not isinstance(text, str) or pd.isna(number):
        return np.nan
    lowered = text.lower()
    if "/month" in lowered or "monthly" in lowered:
        number *= 12
    elif "lpa" in lowered or "lakh" in lowered or "lac" in lowered:
        number *= 100_000
    elif re.search(r"\d\s*k\b", lowered):
        number *= 1_000
    return np.nan if number < 0 else number


def _parse_loan_amount(value: Any) -> float:
    text = _clean_text(value)
    number = _extract_number(text)
    if not isinstance(text, str) or pd.isna(number):
        return np.nan
    lowered = text.lower()
    if re.search(r"(?:\d)\s*(?:l|lac|lakh|lakhs)\b", lowered):
        number *= 100_000
    elif re.search(r"\d\s*k\b", lowered):
        number *= 1_000
    return np.nan if number < 0 else number


def _parse_rate(value: Any) -> float:
    text = _clean_text(value)
    number = _extract_number(text)
    if not isinstance(text, str) or pd.isna(number):
        return np.nan
    if number < 0:
        return np.nan
    if "%" not in text and 0 < number <= 1:
        number *= 100
    return number


def _parse_ratio(value: Any) -> float:
    text = _clean_text(value)
    number = _extract_number(text)
    if not isinstance(text, str) or pd.isna(number):
        return np.nan
    if number < 0:
        return np.nan
    if "%" in text or abs(number) > _RATIO_FRACTION_UPPER_BOUND:
        number /= 100
    return number


def _parse_term(value: Any) -> float:
    text = _clean_text(value)
    number = _extract_number(text)
    if not isinstance(text, str) or pd.isna(number):
        return np.nan
    if re.search(r"\b(?:yr|yrs|year|years)\b", text, flags=re.IGNORECASE):
        number *= 12
    return number if number > 0 else np.nan


def _parse_job_tenure(value: Any) -> float:
    text = _clean_text(value)
    if not isinstance(text, str):
        return np.nan
    lowered = text.lower()
    if lowered.startswith("<"):
        return 0.5
    if lowered.endswith("+"):
        number = _extract_number(text)
        return number if number >= 0 else np.nan
    number = _extract_number(text)
    return number if number >= 0 else np.nan


def _parse_age(value: Any) -> float:
    number = _extract_number(value)
    if pd.isna(number) or number < 0 or number > 120:
        return np.nan
    return number


def _parse_credit_score(value: Any) -> float:
    text = _clean_text(value)
    if not isinstance(text, str) or text.lower() in {"ntc", "new to credit"}:
        return np.nan
    number = _extract_number(text)
    if pd.isna(number) or number < 300 or number > 900:
        return np.nan
    return number


def _parse_nonnegative(value: Any) -> float:
    number = _extract_number(value)
    return number if not pd.isna(number) and number >= 0 else np.nan


def _normalize_category(series: pd.Series, column: str) -> pd.Series:
    cleaned = series.map(_clean_text)
    aliases = _CATEGORY_ALIASES.get(column, {})

    def normalize(value: Any) -> Any:
        if not isinstance(value, str):
            return np.nan
        key = re.sub(r"\s+", " ", value.strip()).lower()
        if column in {"has_cosigner", "has_collateral"}:
            return _YES_NO_MAP.get(key, value.strip())
        if column == "branch_code":
            return re.sub(r"[\s-]+", "", value).upper()
        if column == "employer_category" and key == "not applicable":
            return "Not Applicable"
        if column == "loan_purpose":
            return aliases.get(key, value.strip())
        if aliases:
            return aliases.get(key, value.strip().title())
        return re.sub(r"\s+", " ", value.strip())

    return cleaned.map(normalize)


def parse_application_dates(values: pd.Series) -> pd.Series:
    """Parse observed date formats; slash-separated dates use day/month/year."""
    def parse_one(value: Any) -> pd.Timestamp:
        text = _clean_text(value)
        if not isinstance(text, str):
            return pd.NaT
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
            return pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
        return pd.to_datetime(text, errors="coerce")

    return values.map(parse_one)


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw input values without fitting or imputing from any dataset."""
    if data.empty:
        raise ValueError("Cannot clean an empty dataset.")
    cleaned = data.copy()
    if "credit_score" in cleaned:
        no_history = (
            cleaned["credit_score"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin({"ntc", "new to credit"})
        )
        cleaned["credit_score_no_history"] = no_history.astype(float)

    numeric_parsers = {
        "age": _parse_age,
        "dependents": _parse_nonnegative,
        "years_at_current_job": _parse_job_tenure,
        "annual_income": _parse_annual_income,
        "years_at_residence": _parse_nonnegative,
        "loan_amount": _parse_loan_amount,
        "loan_term_months": _parse_term,
        "interest_rate": _parse_rate,
        "credit_score": _parse_credit_score,
        "num_credit_lines": _parse_nonnegative,
        "existing_monthly_emi": _parse_nonnegative,
        "late_payments_last_2y": _parse_nonnegative,
        "credit_utilization": _parse_ratio,
        "hard_inquiries_6m": _parse_nonnegative,
        "bank_relationship_months": _parse_nonnegative,
        "debt_to_income": _parse_ratio,
    }
    for column, parser in numeric_parsers.items():
        if column in cleaned:
            cleaned[column] = cleaned[column].map(parser).astype(float)

    for column in _CATEGORICAL_COLUMNS:
        if column in cleaned:
            cleaned[column] = _normalize_category(cleaned[column], column)

    if "application_date" in cleaned:
        dates = parse_application_dates(cleaned["application_date"])
        cleaned["application_year"] = dates.dt.year.astype(float)
        cleaned["application_month"] = dates.dt.month.astype(float)
        cleaned = cleaned.drop(columns="application_date")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    validate_cleaned_dtypes(cleaned)
    return cleaned


def validate_cleaned_dtypes(data: pd.DataFrame) -> None:
    """Raise if normalized numeric fields still have unexpected non-numeric types."""
    for column in _NUMERIC_COLUMNS:
        if column in data and not pd.api.types.is_numeric_dtype(data[column]):
            raise TypeError(
                f"Cleaned numeric feature {column!r} has unexpected dtype "
                f"{data[column].dtype}."
            )
        if column in data and np.isinf(data[column].dropna()).any():
            raise ValueError(f"Cleaned numeric feature {column!r} contains infinity.")
    if TARGET_COLUMN in data and not pd.api.types.is_numeric_dtype(data[TARGET_COLUMN]):
        raise TypeError(
            f"Cleaned target {TARGET_COLUMN!r} has unexpected dtype "
            f"{data[TARGET_COLUMN].dtype}."
        )


def normalize_target(target: pd.Series) -> pd.Series:
    """Map documented target tokens to 0/1; preserve missing labels as NaN."""
    values = target.map(_clean_text)
    normalized = values.map(
        lambda value: _TARGET_MAP.get(value.lower()) if isinstance(value, str) else np.nan
    )
    unknown = values.notna() & normalized.isna()
    if unknown.any():
        tokens = values.loc[unknown].astype(str).drop_duplicates().tolist()
        raise ValueError(f"Unrecognized defaulted target values: {tokens[:10]}")
    return normalized.astype("float64")


def clean_training_data(
    data: pd.DataFrame,
    *,
    drop_exact_duplicates: bool = True,
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """Clean training rows, retain the target, and optionally omit unusable labels."""
    if data.empty:
        raise ValueError("Training dataset is empty.")
    if TARGET_COLUMN not in data:
        raise ValueError(f"Training dataset must contain target column {TARGET_COLUMN!r}.")
    cleaned = data.copy()
    if drop_exact_duplicates:
        cleaned = cleaned.drop_duplicates(keep="first").copy()
    cleaned[TARGET_COLUMN] = normalize_target(cleaned[TARGET_COLUMN])
    cleaned = clean_features(cleaned)
    if drop_unlabeled:
        cleaned = cleaned.loc[cleaned[TARGET_COLUMN].notna()].copy()
    return cleaned.reset_index(drop=True)


def clean_inference_data(
    data: pd.DataFrame,
    *,
    allow_scenario_column: bool = True,
) -> pd.DataFrame:
    """Clean test/scenario rows without target information or learned imputation."""
    if data.empty:
        raise ValueError("Inference dataset is empty.")
    if TARGET_COLUMN in data:
        raise ValueError(
            f"Inference data must not contain target column {TARGET_COLUMN!r}."
        )
    if SCENARIO_COLUMN in data and not allow_scenario_column:
        raise ValueError("Scenario narrative column is not valid for this dataset.")
    return clean_features(data)


def get_predictor_columns(data: pd.DataFrame) -> list[str]:
    """Return predictors after excluding targets, IDs, and known leakage fields."""
    return [
        column
        for column in data.columns
        if column not in EXCLUDED_FEATURE_REASONS
    ]


def validate_dataset_schema(
    data: pd.DataFrame,
    role: str,
    *,
    allow_extra_columns: tuple[str, ...] = (),
) -> list[str]:
    """Raise for empty data, missing required fields, or unexpected columns."""
    if data.empty:
        raise ValueError(f"{role} dataset is empty.")
    role = role.lower()
    if role == "train":
        required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
        allowed_extra = set(allow_extra_columns)
    elif role == "test":
        required = set(FEATURE_COLUMNS)
        allowed_extra = set(allow_extra_columns)
    elif role == "scenario":
        required = set(FEATURE_COLUMNS) | {SCENARIO_COLUMN}
        allowed_extra = set(allow_extra_columns)
    else:
        raise ValueError("role must be 'train', 'test', or 'scenario'.")

    missing = sorted(required - set(data.columns))
    unexpected = sorted(set(data.columns) - required - allowed_extra)
    if missing or unexpected:
        raise ValueError(
            f"{role} schema mismatch; missing={missing}, unexpected={unexpected}."
        )
    if role != "train" and TARGET_COLUMN in data:
        raise ValueError(f"{role} data must not contain target {TARGET_COLUMN!r}.")
    if role == "train" and TARGET_COLUMN not in data:
        raise ValueError(f"Training data must contain target {TARGET_COLUMN!r}.")
    return sorted(required)


def validate_training_data(
    data: pd.DataFrame,
    *,
    require_complete_target: bool = True,
    require_no_duplicates: bool = True,
) -> list[str]:
    """Validate training target, duplicate state, and feature/missing-value status."""
    if data.empty:
        raise ValueError("Training dataset is empty.")
    if TARGET_COLUMN not in data:
        raise ValueError(f"Training dataset must contain target column {TARGET_COLUMN!r}.")
    if require_complete_target and data[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Training target contains {int(data[TARGET_COLUMN].isna().sum())} missing labels."
        )
    if require_no_duplicates and data.duplicated().any():
        raise ValueError(
            f"Training dataset contains {int(data.duplicated().sum())} duplicate rows."
        )
    normalized = normalize_target(data[TARGET_COLUMN])
    if normalized.notna().sum() != data[TARGET_COLUMN].notna().sum():
        raise ValueError("Training target contains values that cannot be mapped to 0/1.")
    missing_features = int(data.drop(columns=TARGET_COLUMN).isna().sum().sum())
    return [
        f"Feature values missing: {missing_features}; they are retained for pipeline imputation."
    ]


def validate_inference_data(
    data: pd.DataFrame,
    *,
    expected_features: list[str] | None = None,
    allow_scenario_column: bool = True,
) -> list[str]:
    """Validate inference schema and report, but do not impute, feature missingness."""
    if data.empty:
        raise ValueError("Inference dataset is empty.")
    if TARGET_COLUMN in data:
        raise ValueError(f"Inference data must not contain target {TARGET_COLUMN!r}.")
    allowed = {SCENARIO_COLUMN} if allow_scenario_column else set()
    unexpected = sorted(set(data.columns) - set(expected_features or data.columns) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected inference columns: {unexpected}")
    if expected_features is not None:
        missing = sorted(set(expected_features) - set(data.columns))
        if missing:
            raise ValueError(f"Missing inference feature columns: {missing}")
    missing_features = int(data.drop(columns=list(allowed & set(data.columns))).isna().sum().sum())
    return [
        f"Feature values missing: {missing_features}; they are retained for pipeline imputation."
    ]


def validate_dataset_bundle(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> dict[str, list[str]]:
    """Validate train/test/scenario roles and matching raw feature schemas."""
    validate_dataset_schema(training, "train")
    validate_dataset_schema(testing, "test")
    validate_dataset_schema(scenarios, "scenario")
    train_features = set(training.columns) - {TARGET_COLUMN}
    test_features = set(testing.columns)
    scenario_features = set(scenarios.columns) - {SCENARIO_COLUMN}
    if train_features != test_features or train_features != scenario_features:
        raise ValueError(
            "Train, test, and scenario feature columns do not match after excluding "
            "the training target and scenario description."
        )
    return {
        "training": validate_training_data(
            training, require_complete_target=False, require_no_duplicates=False
        ),
        "testing": validate_inference_data(testing, expected_features=sorted(train_features)),
        "scenarios": validate_inference_data(
            scenarios, expected_features=sorted(train_features), allow_scenario_column=True
        ),
    }


def detect_suspicious_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Report IQR-flagged numeric values without dropping or changing any rows."""
    records: list[dict[str, Any]] = []
    for column in data.select_dtypes(include="number").columns:
        values = data[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        if count:
            records.append(
                {
                    "column": column,
                    "lower_fence": float(lower),
                    "upper_fence": float(upper),
                    "rows_flagged": count,
                    "action": "flag only; do not remove automatically",
                }
            )
    return pd.DataFrame(
        records,
        columns=["column", "lower_fence", "upper_fence", "rows_flagged", "action"],
    )


def validate_raw_application_values(data: pd.DataFrame) -> pd.DataFrame:
    """Return row-level flags for invalid values; does not silently score them."""
    issues: list[dict[str, Any]] = []
    parsers = {
        "age": _parse_age,
        "annual_income": _parse_annual_income,
        "loan_amount": _parse_loan_amount,
        "loan_term_months": _parse_term,
        "interest_rate": _parse_rate,
        "credit_score": _parse_credit_score,
        "dependents": _parse_nonnegative,
        "years_at_current_job": _parse_job_tenure,
        "years_at_residence": _parse_nonnegative,
        "num_credit_lines": _parse_nonnegative,
        "existing_monthly_emi": _parse_nonnegative,
        "late_payments_last_2y": _parse_nonnegative,
        "credit_utilization": _parse_ratio,
        "hard_inquiries_6m": _parse_nonnegative,
        "bank_relationship_months": _parse_nonnegative,
        "debt_to_income": _parse_ratio,
    }
    for column, parser in parsers.items():
        if column not in data:
            continue
        parsed = data[column].map(parser)
        raw_present = data[column].notna()
        if column == "credit_score":
            no_history = (
                data[column].astype("string").str.strip().str.lower().isin(
                    {"ntc", "new to credit"}
                )
            )
            raw_present &= ~no_history
        invalid = raw_present & parsed.isna()
        for index in data.index[invalid]:
            issues.append(
                {
                    "row_index": index,
                    "column": column,
                    "issue": "invalid or unparseable value",
                }
            )
    if "application_date" in data:
        parsed_dates = parse_application_dates(data["application_date"])
        invalid_dates = data["application_date"].notna() & parsed_dates.isna()
        for index in data.index[invalid_dates]:
            issues.append(
                {
                    "row_index": index,
                    "column": "application_date",
                    "issue": "invalid or unparseable date",
                }
            )
    if "loan_amount" in data:
        amount = data["loan_amount"].map(_extract_number)
        negative = amount.notna() & amount.lt(0)
        for index in data.index[negative]:
            issues.append(
                {
                    "row_index": index,
                    "column": "loan_amount",
                    "issue": "negative requested principal",
                }
            )
    return pd.DataFrame(issues, columns=["row_index", "column", "issue"])
