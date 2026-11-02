from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.cleaning import TARGET_COLUMN, clean_inference_data
from src.features import build_model_pipeline, predictor_frame
from src.phase2e_2j import (
    CV_FOLDS,
    SCORING,
    _write_interpretation,
    analyze_thresholds,
    build_model_zoo,
    build_smote_nc_pipeline,
    run_model_zoo,
    tune_top_models,
    validate_prediction_frame,
)

PROJECT = Path(__file__).resolve().parents[1]


def _synthetic_training(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    target = np.tile([0, 0, 0, 1], rows // 4)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 70, rows),
            "annual_income": rng.integers(250_000, 1_500_000, rows),
            "employment_type": np.where(
                np.arange(rows) % 3 == 0, "Salaried", "Self-Employed"
            ),
            "applicant_id": [f"ID{i}" for i in range(rows)],
            TARGET_COLUMN: target,
        }
    )


def test_all_ten_required_estimators_initialize():
    models = build_model_zoo()
    assert len(models) == 10
    assert {
        "DummyClassifier",
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Gaussian Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "HistGradientBoosting",
        "AdaBoost",
        "Support Vector Machine",
    } == set(models)


def test_model_zoo_runs_five_fold_multimetric_cv_on_synthetic_data():
    frame = _synthetic_training()
    training = frame.iloc[:60].copy()
    validation = frame.iloc[60:].copy()
    X_train = predictor_frame(training)
    X_validation = predictor_frame(validation)
    result = run_model_zoo(
        X_train,
        training[TARGET_COLUMN],
        X_validation,
        validation[TARGET_COLUMN],
    )
    assert len(result["comparison"]) >= 9
    assert set(SCORING) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert (result["comparison"]["CV ROC_AUC Std"] >= 0).all()
    assert (result["comparison"]["CV ROC_AUC Mean"].between(0, 1)).all()
    assert all(len(scores) == CV_FOLDS for scores in result["cv_roc_auc"].values())


def test_cv_folds_are_stratified_and_preserve_class_rates():
    y = _synthetic_training(100)[TARGET_COLUMN]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    overall_rate = y.mean()
    for train_index, validation_index in splitter.split(np.zeros(len(y)), y):
        assert abs(y.iloc[train_index].mean() - overall_rate) < 0.02
        assert abs(y.iloc[validation_index].mean() - overall_rate) < 0.02


def test_smote_nc_is_inside_pipeline_and_does_not_mutate_validation():
    frame = _synthetic_training()
    X = predictor_frame(frame)
    y = frame[TARGET_COLUMN]
    X_train, X_validation = X.iloc[:60].copy(), X.iloc[60:].copy()
    y_train = y.iloc[:60].copy()
    validation_before = X_validation.copy(deep=True)
    pipeline = build_smote_nc_pipeline(
        X_train, LogisticRegression(max_iter=100, random_state=42)
    )
    assert pipeline.steps[1][0] == "smote_nc"
    pipeline.fit(X_train, y_train)
    pd.testing.assert_frame_equal(X_validation, validation_before)
    sampled_counts = pipeline.named_steps["smote_nc"].sampling_strategy_
    assert sampled_counts


def test_grid_search_tunes_two_models_using_five_folds():
    frame = _synthetic_training()
    train = frame.iloc[:60]
    X_train = predictor_frame(train)
    comparison = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree"],
            "CV ROC_AUC Mean": [0.7, 0.68],
            "CV ROC_AUC Std": [0.02, 0.03],
            "Validation Fit Seconds": [0.01, 0.01],
        }
    )
    result = tune_top_models(comparison, X_train, train[TARGET_COLUMN])
    assert len(result["results"]) == 2
    assert (result["results"]["Candidates"] > 0).all()
    assert (result["results"]["Tuned CV ROC-AUC"].between(0, 1)).all()
    assert all(
        len([key for key in search.cv_results_ if key.startswith("split") and key.endswith("_test_score")]) == 5
        for search in result["searches"].values()
    )


def test_threshold_costs_and_optimization_are_correct():
    y = np.array([1, 0, 1, 0])
    probability = np.array([0.2, 0.3, 0.8, 0.6])
    table = analyze_thresholds(
        "unit", y, probability, thresholds=np.array([0.05, 0.5, 0.95])
    )
    midpoint = table.loc[table["Threshold"] == 0.5].iloc[0]
    assert midpoint["False Negatives"] == 1
    assert midpoint["False Positives"] == 1
    assert midpoint["Business Cost"] == 115_000
    assert midpoint["Cost per Applicant"] == 28_750
    assert midpoint["Recall"] == 0.5
    assert midpoint["Precision"] == 0.5
    optimal = table.sort_values("Business Cost").iloc[0]
    assert optimal["Business Cost"] <= midpoint["Business Cost"]


def test_global_importance_and_three_requested_scenario_explanations(tmp_path):
    frame = _synthetic_training(80)
    training = frame.iloc[:60]
    validation = frame.iloc[60:]
    X_train, X_validation = predictor_frame(training), predictor_frame(validation)
    y_train, y_validation = training[TARGET_COLUMN], validation[TARGET_COLUMN]
    pipeline = build_model_pipeline(
        X_train, LogisticRegression(max_iter=200, random_state=42)
    ).fit(X_train, y_train)
    raw_scenario = training.iloc[:3].drop(columns=TARGET_COLUMN).copy()
    raw_scenario["applicant_id"] = ["TC01", "TC02", "TC08"]
    raw_scenario["scenario"] = ["prime", "high risk", "self employed"]
    cleaned_scenario = clean_inference_data(raw_scenario)
    importance, coefficients, explanations = _write_interpretation(
        selected_threshold=0.4,
        X_train=X_train,
        y_train=y_train,
        validation_pipeline=pipeline,
        scenario_pipeline=pipeline,
        X_validation=X_validation,
        y_validation=y_validation,
        raw_scenario=raw_scenario,
        cleaned_scenario=cleaned_scenario,
        output=tmp_path,
    )
    assert not importance.empty
    assert not coefficients.empty
    assert explanations["applicant_id"].tolist() == ["TC01", "TC02", "TC08"]
    assert (tmp_path / "permutation_importance.csv").exists()
    assert (tmp_path / "feature_importance.png").exists()
    assert (tmp_path / "logistic_regression_coefficients.csv").exists()
    assert (tmp_path / "scenario_explanations.md").exists()


def test_test_predictions_have_exact_schema_and_all_required_rows():
    prediction_path = PROJECT / "outputs" / "predictions_loan_test.csv"
    assert prediction_path.is_file(), "Run src/train.py before integration-output tests."
    predictions = pd.read_csv(prediction_path)
    source = pd.read_csv(
        PROJECT.parent.parent / "DS_Trainer_Task2_Data" / "loan_test.csv",
        low_memory=False,
    )
    validate_prediction_frame(
        predictions, source["applicant_id"], expected_rows=8_000
    )
    assert predictions.columns.tolist() == [
        "applicant_id",
        "default_probability",
        "default_pred",
    ]


def test_all_scenario_predictions_and_final_model_artifact_exist():
    scenario_path = PROJECT / "outputs" / "scenario_predictions.csv"
    model_path = PROJECT / "models" / "final_model.joblib"
    assert scenario_path.is_file(), "Run src/train.py before integration-output tests."
    assert model_path.is_file()
    scenarios = pd.read_csv(scenario_path)
    assert len(scenarios) == 12
    assert scenarios["applicant_id"].tolist() == [
        f"TC{i:02d}" for i in range(1, 13)
    ]
    assert scenarios["default_pred"].isin([0, 1]).all()
    assert scenarios["default_probability"].between(0, 1).all()
    artifact = joblib.load(model_path)
    assert artifact["metadata"]["test_labels_used"] is False
    assert artifact["metadata"]["decision_threshold"] == artifact["threshold"]


def test_prediction_cli_accepts_named_input_and_output_options(tmp_path):
    source = PROJECT.parent.parent / "DS_Trainer_Task2_Data" / "loan_scenario_cases.csv"
    destination = tmp_path / "scenario_cli_predictions.csv"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "predict.py"),
            "--input",
            str(source),
            "--output",
            str(destination),
        ],
        cwd=PROJECT,
        check=True,
        capture_output=True,
        text=True,
    )

    predictions = pd.read_csv(destination)
    expected = pd.read_csv(PROJECT / "outputs" / "scenario_predictions.csv")
    assert predictions["applicant_id"].tolist() == expected["applicant_id"].tolist()
    assert predictions["default_pred"].tolist() == expected["default_pred"].tolist()
    np.testing.assert_allclose(
        predictions["default_probability"],
        expected["default_probability"],
        rtol=0,
        atol=1e-12,
    )
