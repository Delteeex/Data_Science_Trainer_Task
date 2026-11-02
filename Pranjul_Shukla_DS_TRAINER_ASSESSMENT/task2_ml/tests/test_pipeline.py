from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    EXCLUDED_FEATURE_REASONS,
    TARGET_COLUMN,
    clean_features,
    clean_inference_data,
    clean_training_data,
    get_predictor_columns,
    load_datasets,
    validate_dataset_bundle,
    validate_dataset_schema,
    validate_inference_data,
    validate_raw_application_values,
    validate_training_data,
)
from src.features import build_model_pipeline, build_preprocessor, predictor_frame
from src.train import (
    BREAK_EVEN_THRESHOLD,
    THRESHOLD_GRID,
    _verify_and_save_model,
    analyze_thresholds,
    candidate_estimators,
    evaluate_candidates,
    load_training_csv,
    make_stratified_split,
    prepare_datasets,
)


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def tiny_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, np.nan, 40, 35],
            "annual_income": [600_000, 900_000, np.nan, 750_000],
            "gender": ["Female", "Male", None, "Female"],
            "loan_purpose": ["Home Renovation", "Personal", "Personal", "Education"],
            "applicant_id": ["A1", "A2", "A3", "A4"],
            "recovery_calls_made": [0, 5, 0, 2],
            "defaulted": [0, 1, 0, 1],
        }
    )


def test_loads_all_three_original_datasets():
    datasets = load_datasets(ROOT / "DS_Trainer_Task2_Data")
    assert set(datasets) == {"train", "test", "scenario"}
    assert datasets["train"].shape == (32_600, 35)
    assert datasets["test"].shape == (8_000, 34)
    assert datasets["scenario"].shape == (12, 35)
    assert TARGET_COLUMN in datasets["train"]
    assert TARGET_COLUMN not in datasets["test"]
    assert TARGET_COLUMN not in datasets["scenario"]


def test_cleaning_preserves_target_and_drops_exact_duplicates_and_unlabeled():
    raw = pd.DataFrame(
        {
            "age": ["34", "34", "35"],
            "loan_amount": ["₹400,000", "₹400,000", "-250000"],
            "defaulted": ["Yes", "Yes", None],
        }
    )
    cleaned = clean_training_data(raw)
    assert TARGET_COLUMN in cleaned
    assert cleaned[TARGET_COLUMN].tolist() == [1.0]
    assert len(cleaned) == 1
    assert cleaned.loc[0, "loan_amount"] == 400_000


def test_missing_values_are_left_for_pipeline_imputation(tiny_features):
    features = predictor_frame(tiny_features)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    assert transformed.shape[0] == len(features)
    assert np.isfinite(transformed.toarray() if hasattr(transformed, "toarray") else transformed).all()


def test_categorical_encoding_and_numeric_scaling(tiny_features):
    features = predictor_frame(tiny_features)
    preprocessor = build_preprocessor(features)
    transformed = preprocessor.fit_transform(features)
    names = preprocessor.get_feature_names_out().tolist()
    assert any("gender_" in name for name in names)
    assert any("age" in name for name in names)
    assert transformed.shape[1] == len(names)


def test_unknown_categories_are_ignored_without_feature_mismatch(tiny_features):
    train_features = predictor_frame(tiny_features)
    preprocessor = build_preprocessor(train_features)
    preprocessor.fit(train_features)
    incoming = train_features.iloc[[0]].copy()
    incoming.loc[:, "loan_purpose"] = "Crypto Investment"
    transformed = preprocessor.transform(incoming)
    assert transformed.shape[1] == len(preprocessor.get_feature_names_out())


def test_mixed_percentage_and_fraction_ratio_formats_are_normalized():
    raw = pd.DataFrame(
        {"credit_utilization": ["0.453", "53.4%", "1.05", "28.1"]}
    )
    cleaned = clean_features(raw)
    assert cleaned["credit_utilization"].tolist() == pytest.approx(
        [0.453, 0.534, 1.05, 0.281]
    )


def test_no_credit_history_is_distinguished_from_missing_score():
    raw = pd.DataFrame(
        {"credit_score": ["NTC", "New to Credit", None, "720"]}
    )
    cleaned = clean_features(raw)
    assert cleaned["credit_score_no_history"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert cleaned.loc[0:2, "credit_score"].isna().all()
    assert cleaned.loc[3, "credit_score"] == 720


def test_invalid_application_values_are_flagged_before_scoring():
    raw = pd.DataFrame(
        {"loan_amount": ["-250000", "400000"], "age": ["abc", "34"]}
    )
    issues = validate_raw_application_values(raw)
    assert ((issues["row_index"] == 0) & (issues["column"] == "loan_amount")).any()
    assert ((issues["row_index"] == 0) & (issues["column"] == "age")).any()


def test_category_aliases_are_normalized_consistently():
    raw = pd.DataFrame(
        {
            "gender": ["M", "male", "FEMALE"],
            "city_tier": ["T1", "Metro", "tier2"],
            "has_cosigner": ["FALSE", "no", "Y"],
        }
    )
    cleaned = clean_features(raw)
    assert cleaned["gender"].tolist() == ["Male", "Male", "Female"]
    assert cleaned["city_tier"].tolist() == ["Tier 1", "Tier 1", "Tier 2"]
    assert cleaned["has_cosigner"].tolist() == ["No", "No", "Yes"]


def test_train_test_predictor_consistency(tiny_features):
    cleaned = clean_training_data(tiny_features)
    test = clean_inference_data(tiny_features.drop(columns=TARGET_COLUMN))
    assert get_predictor_columns(cleaned) == get_predictor_columns(test)
    assert predictor_frame(cleaned).columns.tolist() == predictor_frame(test).columns.tolist()


def test_scenario_dataset_compatibility_and_flags_invalid_row():
    datasets = load_datasets(ROOT / "DS_Trainer_Task2_Data")
    validate_dataset_bundle(
        datasets["train"], datasets["test"], datasets["scenario"]
    )
    prepared = prepare_datasets(ROOT / "DS_Trainer_Task2_Data")
    train_features = predictor_frame(prepared["train"])
    scenarios = predictor_frame(prepared["scenario"])
    preprocessor = build_preprocessor(train_features)
    preprocessor.fit(train_features)
    transformed = preprocessor.transform(scenarios)
    assert len(prepared["scenario"]) == 12
    assert transformed.shape[0] == 12
    issues = validate_raw_application_values(datasets["scenario"])
    assert ((issues["column"] == "loan_amount") & (issues["row_index"] == 10)).any()


def test_target_and_leakage_fields_are_excluded_from_predictors(tiny_features):
    predictors = get_predictor_columns(tiny_features)
    assert TARGET_COLUMN not in predictors
    assert "recovery_calls_made" not in predictors
    assert "applicant_id" not in predictors
    assert "recovery_calls_made" in EXCLUDED_FEATURE_REASONS
    assert "interest_rate" not in predictors
    assert "interest_rate" in EXCLUDED_FEATURE_REASONS


def test_training_validation_detects_missing_labels_and_duplicates():
    missing_target = pd.DataFrame({"value": [1], TARGET_COLUMN: [np.nan]})
    with pytest.raises(ValueError, match="missing labels"):
        validate_training_data(missing_target)
    duplicate = pd.DataFrame({"value": [1, 1], TARGET_COLUMN: [0, 0]})
    with pytest.raises(ValueError, match="duplicate rows"):
        validate_training_data(duplicate)


def test_empty_datasets_and_unexpected_columns_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        clean_features(pd.DataFrame())
    with pytest.raises(ValueError, match="empty"):
        validate_inference_data(pd.DataFrame())
    with pytest.raises(ValueError, match="unexpected"):
        validate_dataset_schema(
            pd.DataFrame({"unknown": [1]}), "test"
        )


def test_stratified_split_is_reproducible(tiny_features):
    training = clean_training_data(tiny_features)
    first = make_stratified_split(training, test_size=0.5)
    second = make_stratified_split(training, test_size=0.5)
    pd.testing.assert_frame_equal(first["X_train"], second["X_train"])
    pd.testing.assert_series_equal(first["y_validation"], second["y_validation"])


def test_threshold_analysis_uses_default_as_positive_and_business_costs():
    result = analyze_thresholds(
        "synthetic",
        np.array([1, 0, 1, 0]),
        np.array([0.2, 0.3, 0.8, 0.6]),
        thresholds=np.array([0.5]),
    ).iloc[0]
    assert result["False Negatives"] == 1
    assert result["False Positives"] == 1
    assert result["Business Cost"] == 115_000
    assert result["Recall"] == pytest.approx(0.5)
    assert result["Precision"] == pytest.approx(0.5)
    assert result["Cost per Applicant"] == pytest.approx(28_750)
    assert 0.5 in THRESHOLD_GRID
    assert BREAK_EVEN_THRESHOLD in THRESHOLD_GRID


def test_baseline_models_evaluate_with_fold_fitted_preprocessing():
    training = pd.DataFrame(
        {
            "age": [21, 24, 28, 31, 35, 38, 42, 45, 50, 54, 59, 63],
            "annual_income": [
                300_000, 450_000, 500_000, 600_000, 700_000, 800_000,
                900_000, 1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000,
            ],
            "employment_type": [
                "Salaried", "Self-Employed", "Salaried", "Salaried",
                "Self-Employed", "Salaried", "Salaried", "Self-Employed",
                "Salaried", "Salaried", "Self-Employed", "Salaried",
            ],
            "applicant_id": [f"ID{i}" for i in range(12)],
            "defaulted": [0, 1] * 6,
        }
    )
    train_X = predictor_frame(training.iloc[:8])
    validation_X = predictor_frame(training.iloc[8:])
    result = evaluate_candidates(
        train_X,
        training["defaulted"].iloc[:8],
        validation_X,
        training["defaulted"].iloc[8:],
        estimators={
            "Logistic Regression": candidate_estimators()["Logistic Regression"],
            "Decision Tree": candidate_estimators()["Decision Tree"],
        },
    )
    assert set(result["comparison"]["Model"]) == {
        "Logistic Regression",
        "Decision Tree",
    }
    assert TARGET_COLUMN not in result["fitted_models"]["Logistic Regression"].feature_names_in_
    assert all(
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all()
        for probabilities in result["validation_probabilities"].values()
    )


def test_phase3_training_loader_reads_only_training_csv(tmp_path):
    project_dir = tmp_path / "workspace" / "candidate" / "task2_ml"
    data_dir = project_dir.parent.parent / "DS_Trainer_Task2_Data"
    data_dir.mkdir(parents=True)
    training = pd.DataFrame({"age": [30], TARGET_COLUMN: [1]})
    training.to_csv(data_dir / "loan_train.csv", index=False)
    (data_dir / "loan_test.csv").write_text("intentionally invalid csv", encoding="utf-8")
    (data_dir / "loan_scenario_cases.csv").write_text(
        "intentionally invalid csv", encoding="utf-8"
    )
    loaded = load_training_csv(project_dir)
    pd.testing.assert_frame_equal(loaded, training)


def test_verified_model_artifact_round_trip_preserves_threshold_and_schema(
    tiny_features, tmp_path
):
    from sklearn.linear_model import LogisticRegression

    features = predictor_frame(tiny_features)
    pipeline = build_model_pipeline(features, LogisticRegression(random_state=42))
    pipeline.fit(features, tiny_features[TARGET_COLUMN])
    path = _verify_and_save_model(
        pipeline,
        model_dir=tmp_path / "models",
        input_features=features.columns.tolist(),
        threshold=0.37,
        metadata={"model_name": "Logistic Regression"},
        validation_features=features,
    )
    import joblib

    artifact = joblib.load(path)
    assert artifact["threshold"] == pytest.approx(0.37)
    assert artifact["input_features"] == features.columns.tolist()
    assert np.isfinite(artifact["pipeline"].predict_proba(features)).all()
