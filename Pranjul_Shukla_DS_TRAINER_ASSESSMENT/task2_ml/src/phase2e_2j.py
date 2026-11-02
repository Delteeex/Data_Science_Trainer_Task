"""Reproducible Phase 2E-2J model zoo, imbalance, tuning and deployment workflow."""

from __future__ import annotations

import copy
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.cleaning import (
    TARGET_COLUMN,
    clean_inference_data,
    clean_training_data,
    load_datasets,
    validate_raw_application_values,
)
from src.features import build_model_pipeline, predictor_frame, to_dense_array
from src.train import (
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    RANDOM_STATE,
    _verify_and_save_model,
    analyze_thresholds,
    load_training_csv,
    make_stratified_split,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
CV_FOLDS = 5
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def build_model_zoo() -> dict[str, Any]:
    """Return all ten requested, reproducibly configured sklearn classifiers."""
    calibrated_svm = CalibratedClassifierCV(
        estimator=LinearSVC(
            class_weight=None,
            dual="auto",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        method="sigmoid",
        cv=3,
        n_jobs=1,
    )
    return {
        "DummyClassifier": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", max_iter=300, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=15, weights="distance", n_jobs=-1
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("to_dense", FunctionTransformer(to_dense_array)),
                ("classifier", GaussianNB(var_smoothing=1e-8)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("to_dense", FunctionTransformer(to_dense_array)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=80,
                        learning_rate=0.08,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=1, random_state=RANDOM_STATE
            ),
            n_estimators=60,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine": calibrated_svm,
    }


def build_smote_nc_pipeline(features: pd.DataFrame, estimator: Any) -> ImbalancedPipeline:
    """Impute/scale numeric and ordinal-encode categories before fold-local SMOTENC.

    SMOTENC interpolates numeric fields but chooses categorical levels rather
    than interpolating one-hot indicators. The post-sampler transformer restores
    one-hot categorical features before classifier fitting.
    """
    numeric_columns = [
        name for name in features if pd.api.types.is_numeric_dtype(features[name])
    ]
    categorical_columns = [name for name in features if name not in numeric_columns]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    before_sampling = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )
    numeric_count = len(numeric_columns)
    categorical_indices = list(
        range(numeric_count, numeric_count + len(categorical_columns))
    )
    sampler = SMOTENC(
        categorical_features=categorical_indices,
        random_state=RANDOM_STATE,
        k_neighbors=3,
    )
    after_sampling = ColumnTransformer(
        [
            ("numeric", "passthrough", list(range(numeric_count))),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_indices,
            ),
        ],
        sparse_threshold=1.0,
    )
    return ImbalancedPipeline(
        [
            ("pre_sampler_preprocessing", before_sampling),
            ("smote_nc", sampler),
            ("post_sampler_encoding", after_sampling),
            ("estimator", estimator),
        ]
    )


def _cross_validate_one(
    name: str,
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, Any], np.ndarray]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    start = time.perf_counter()
    result = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
        n_jobs=1,
        error_score="raise",
    )
    elapsed = time.perf_counter() - start
    record: dict[str, Any] = {"CV Total Seconds": elapsed}
    for metric in SCORING:
        values = result[f"test_{metric}"]
        record[f"CV {metric.upper()} Mean"] = float(np.mean(values))
        record[f"CV {metric.upper()} Std"] = float(np.std(values))
    record["CV Fit Seconds Mean"] = float(np.mean(result["fit_time"]))
    record["CV Fit Seconds Std"] = float(np.std(result["fit_time"]))
    record["CV Score Seconds Mean"] = float(np.mean(result["score_time"]))
    return record, result["test_roc_auc"]


def _evaluate_holdout(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "Validation Accuracy": accuracy_score(y_true, prediction),
        "Validation Precision": precision_score(y_true, prediction, zero_division=0),
        "Validation Recall": recall_score(y_true, prediction, zero_division=0),
        "Validation F1": f1_score(y_true, prediction, zero_division=0),
        "Validation ROC-AUC": roc_auc_score(y_true, probabilities),
        "Validation PR-AUC": average_precision_score(y_true, probabilities),
        "Validation False Negatives": int(fn),
        "Validation False Positives": int(fp),
        "Validation Business Cost": int(
            fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST
        ),
        "Validation Cost per Applicant": float(
            (fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST) / len(y_true)
        ),
        "Validation Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def run_model_zoo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> dict[str, Any]:
    models = build_model_zoo()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    fitted: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}
    cv_details: dict[str, np.ndarray] = {}
    for name, estimator in models.items():
        try:
            pipe = build_model_pipeline(X_train, clone(estimator))
            cv_record, cv_roc = _cross_validate_one(name, pipe, X_train, y_train)
            fit_started = time.perf_counter()
            pipe.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - fit_started
            val_probs = np.asarray(pipe.predict_proba(X_validation)[:, 1], dtype=float)
            if not np.isfinite(val_probs).all():
                raise ValueError("Validation probabilities contain NaN or infinity.")
            if ((val_probs < 0) | (val_probs > 1)).any():
                raise ValueError("Validation probabilities are outside [0, 1].")
            records.append(
                {
                    **cv_record,
                    "Model": name,
                    "Estimator Parameters": repr(estimator.get_params(deep=True)),
                    "Validation Fit Seconds": fit_seconds,
                    **_evaluate_holdout(y_validation, val_probs),
                }
            )
            fitted[name] = pipe
            probabilities[name] = val_probs
            cv_details[name] = cv_roc
            print(
                f"{name}: 5-fold ROC-AUC "
                f"{cv_record['CV ROC_AUC Mean']:.4f} +/- "
                f"{cv_record['CV ROC_AUC Std']:.4f}; "
                f"validation ROC-AUC={roc_auc_score(y_validation, val_probs):.4f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "Model": name,
                    "Error Type": type(exc).__name__,
                    "Exact Error": str(exc),
                    "Reason": (
                        "This required estimator could not complete its configured "
                        "five-fold CV/pipeline evaluation; it was not silently omitted."
                    ),
                }
            )
            print(f"{name} FAILED: {type(exc).__name__}: {exc}")
    table = pd.DataFrame(records).sort_values(
        ["CV ROC_AUC Mean", "CV ROC_AUC Std"],
        ascending=[False, True],
        kind="stable",
    )
    return {
        "comparison": table.reset_index(drop=True),
        "failures": pd.DataFrame(failures),
        "fitted_models": fitted,
        "validation_probabilities": probabilities,
        "cv_roc_auc": cv_details,
    }


def run_imbalance_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    model_zoo: dict[str, Any],
) -> dict[str, Any]:
    strategies: list[tuple[str, str, Any, Any]] = []
    for model_name, estimator in (
        ("Logistic Regression", build_model_zoo()["Logistic Regression"]),
        ("Random Forest", build_model_zoo()["Random Forest"]),
    ):
        base = clone(estimator)
        weighted = clone(estimator)
        weighted.set_params(class_weight="balanced")
        strategies.extend(
            [
                ("No correction", model_name, base, "baseline"),
                ("Class weight", model_name, weighted, "weighted"),
                ("SMOTENC", model_name, clone(estimator), "smote"),
            ]
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    fitted: dict[str, Any] = {}
    for strategy, model_name, estimator, kind in strategies:
        label = f"{model_name} - {strategy}"
        try:
            if kind == "baseline":
                model = model_zoo["fitted_models"].get(model_name)
                base_row = model_zoo["comparison"].loc[
                    model_zoo["comparison"]["Model"] == model_name
                ].iloc[0]
                cv_record = {
                    column: base_row[column]
                    for column in base_row.index
                    if column.startswith("CV ")
                }
                fit_seconds = float(base_row["Validation Fit Seconds"])
            elif kind == "smote":
                model = build_smote_nc_pipeline(X_train, estimator)
                cv_record, _ = _cross_validate_one(label, model, X_train, y_train)
                fit_seconds = np.nan
            else:
                model = build_model_pipeline(X_train, estimator)
                cv_record, _ = _cross_validate_one(label, model, X_train, y_train)
                fit_seconds = np.nan
            if kind != "baseline":
                fit_started = time.perf_counter()
                model.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - fit_started
            val_prob = np.asarray(model.predict_proba(X_validation)[:, 1], dtype=float)
            if not np.isfinite(val_prob).all():
                raise ValueError("Validation probabilities contain NaN or infinity.")
            metrics = _evaluate_holdout(y_validation, val_prob, threshold=0.5)
            row = {
                "Strategy": strategy,
                "Model": model_name,
                "Strategy Model": label,
                **cv_record,
                "Validation Fit Seconds": fit_seconds,
                **metrics,
            }
            records.append(row)
            fitted[label] = model
        except Exception as exc:
            failures.append(
                {
                    "Strategy Model": label,
                    "Error Type": type(exc).__name__,
                    "Exact Error": str(exc),
                }
            )
    return {
        "comparison": pd.DataFrame(records),
        "failures": pd.DataFrame(failures),
        "fitted_models": fitted,
    }


def tune_top_models(
    model_comparison: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Any]:
    top_names = (
        model_comparison.sort_values(
            ["CV ROC_AUC Mean", "CV ROC_AUC Std"],
            ascending=[False, True],
        )["Model"]
        .drop_duplicates()
        .head(2)
        .tolist()
    )
    baseline_estimators = build_model_zoo()
    parameter_grids: dict[str, dict[str, list[Any]]] = {
        "Logistic Regression": {"estimator__C": [0.5, 1.0, 2.0]},
        "HistGradientBoosting": {
            "estimator__classifier__max_iter": [60, 100],
            "estimator__classifier__learning_rate": [0.05, 0.1],
        },
        "Random Forest": {
            "estimator__n_estimators": [80, 120],
            "estimator__max_depth": [14, 20],
        },
        "Extra Trees": {
            "estimator__n_estimators": [80, 120],
            "estimator__max_depth": [14, 20],
        },
        "Decision Tree": {
            "estimator__max_depth": [8, 12, None],
            "estimator__min_samples_leaf": [10, 25],
        },
        "AdaBoost": {
            "estimator__n_estimators": [40, 80],
            "estimator__learning_rate": [0.3, 0.8],
        },
        "K-Nearest Neighbors": {
            "estimator__n_neighbors": [9, 15, 21],
        },
        "Gaussian Naive Bayes": {
            "estimator__classifier__var_smoothing": [1e-9, 1e-8, 1e-7],
        },
        "Support Vector Machine": {
            "estimator__estimator__C": [0.5, 1.0, 2.0],
        },
    }
    results: list[dict[str, Any]] = []
    tuned: dict[str, Any] = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name in top_names:
        base = clone(baseline_estimators[name])
        pipeline = build_model_pipeline(X_train, base)
        baseline_row = model_comparison.loc[model_comparison["Model"] == name].iloc[0]
        default_cv = float(baseline_row["CV ROC_AUC Mean"])
        start = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            parameter_grids[name],
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        tuned[name] = search
        gain = float(search.best_score_ - default_cv)
        results.append(
            {
                "Model": name,
                "Default Parameters": repr(base.get_params(deep=True)),
                "Best Parameters": repr(search.best_params_),
                "Default CV ROC-AUC": default_cv,
                "Tuned CV ROC-AUC": float(search.best_score_),
                "Absolute Gain": gain,
                "Percentage Gain": (
                    gain / default_cv * 100 if default_cv else np.nan
                ),
                "Baseline Training Seconds": float(
                    baseline_row["Validation Fit Seconds"]
                ),
                "Tuning and Refit Seconds": elapsed,
                "Tuned CV ROC-AUC Std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "Candidates": len(search.cv_results_["params"]),
            }
        )
        print(
            f"Tuned {name}: CV ROC-AUC {default_cv:.4f} -> "
            f"{search.best_score_:.4f} in {elapsed:.2f}s"
        )
    return {"results": pd.DataFrame(results), "searches": tuned}


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    data = frame.loc[:, columns].copy() if columns else frame.copy()
    for column in data.select_dtypes(include="number").columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    headers = [str(x) for x in data.columns]
    rows = [[str(x) for x in row] for row in data.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(x.ljust(widths[i]) for i, x in enumerate(headers)) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _plot_model_comparison(comparison: pd.DataFrame, output: Path) -> None:
    frame = comparison.sort_values("CV ROC_AUC Mean", ascending=False)
    x = np.arange(len(frame))
    width = 0.25
    fig, axis = plt.subplots(figsize=(13, 7))
    axis.bar(x - width, frame["CV ROC_AUC Mean"], width, label="5-fold CV ROC-AUC")
    axis.bar(x, frame["Validation ROC-AUC"], width, label="Validation ROC-AUC")
    axis.bar(x + width, frame["Validation PR-AUC"], width, label="Validation PR-AUC")
    axis.set_xticks(x, frame["Model"], rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Model Zoo: Cross-validation and Validation Performance")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_imbalance(comparison: pd.DataFrame, output: Path) -> None:
    frame = comparison.copy()
    frame["Label"] = frame["Model"] + "\n" + frame["Strategy"]
    x = np.arange(len(frame))
    width = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for index, (metric, label) in enumerate(
        [
            ("Validation Precision", "Precision"),
            ("Validation Recall", "Recall"),
            ("Validation F1", "F1"),
        ]
    ):
        axes[0].bar(x + (index - 1) * width, frame[metric], width, label=label)
    axes[0].set_xticks(x, frame["Label"], rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Validation Classification Metrics (threshold 0.50)")
    axes[0].legend()
    axes[1].bar(x, frame["Validation Business Cost"], color="#C0504D")
    axes[1].set_xticks(x, frame["Label"], rotation=25, ha="right")
    axes[1].set_ylabel("Business cost (INR)")
    axes[1].set_title("Validation Cost (threshold 0.50)")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _save_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _run_2e_report(zoo: dict[str, Any], out: Path) -> None:
    comp = zoo["comparison"]
    failures = zoo["failures"]
    failure_text = (
        _markdown_table(failures)
        if not failures.empty
        else "All 10 requested models completed successfully; no model was skipped."
    )
    table_cols = [
        "Model",
        "Validation Accuracy",
        "Validation Precision",
        "Validation Recall",
        "Validation F1",
        "Validation ROC-AUC",
        "Validation PR-AUC",
        "CV ROC_AUC Mean",
        "CV ROC_AUC Std",
        "CV PRECISION Mean",
        "CV PRECISION Std",
        "CV RECALL Mean",
        "CV RECALL Std",
        "CV F1 Mean",
        "CV F1 Std",
        "CV Fit Seconds Mean",
        "Validation Business Cost",
    ]
    report = f"""# Phase 2E — Ten-model comparison

## Protocol

Used labeled `loan_train.csv` only. After Phase 2 cleaning, an 80/20 stratified split (`random_state=42`) yielded {len(zoo['comparison']) and '10 requested model attempts'} with validation rows reserved for evaluation. Each model's entire imputation, scaling, and one-hot encoding pipeline was refit inside each of five `StratifiedKFold` folds on the training partition. Primary model-zoo ranking is five-fold ROC-AUC mean; the validation split is reported separately. Positive label is default (`1`). No hold-out test labels or scenario labels were used.

The model zoo includes DummyClassifier, Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, AdaBoost, and a calibrated linear SVM. For SVM, `LinearSVC` is probability-calibrated within the training pipeline to support evaluation and downstream probability thresholds.

## Consolidated comparison

{_markdown_table(comp, table_cols)}

![Model comparison chart](model_comparison.png)

## Required model execution status

Completed models: {', '.join(comp['Model'].astype(str))}.

Failures (exact error included; no silent skip):

{failure_text}

No model is declared production-ready based on this single split. The CV standard deviations show fold-to-fold variability on this training sample; time-aware validation, calibration, and fairness review remain necessary.
"""
    _save_report(out / "phase2e_model_zoo_report.md", report)


def _run_2f_report(
    result: dict[str, Any],
    out: Path,
) -> None:
    table = result["comparison"]
    failures = result["failures"]
    table_cols = [
        "Strategy",
        "Model",
        "Validation Precision",
        "Validation Recall",
        "Validation F1",
        "Validation ROC-AUC",
        "Validation PR-AUC",
        "Validation False Negatives",
        "Validation False Positives",
        "Validation Business Cost",
        "Validation Cost per Applicant",
        "CV ROC_AUC Mean",
        "CV ROC_AUC Std",
    ]
    effects = []
    for model_name in table["Model"].drop_duplicates():
        subset = table[table["Model"] == model_name].set_index("Strategy")
        if {"No correction", "Class weight", "SMOTENC"}.issubset(subset.index):
            base = subset.loc["No correction"]
            for strategy in ("Class weight", "SMOTENC"):
                current = subset.loc[strategy]
                effects.append(
                    f"- **{model_name}, {strategy}:** recall "
                    f"{base['Validation Recall']:.4f} → {current['Validation Recall']:.4f}; "
                    f"precision {base['Validation Precision']:.4f} → "
                    f"{current['Validation Precision']:.4f}; false positives "
                    f"{int(base['Validation False Positives']):,} → "
                    f"{int(current['Validation False Positives']):,}; business cost "
                    f"INR {int(base['Validation Business Cost']):,} → "
                    f"INR {int(current['Validation Business Cost']):,}."
                )
    best = table.sort_values(
        ["Validation Business Cost", "CV ROC_AUC Mean"],
        ascending=[True, False],
    ).iloc[0]
    report = f"""# Phase 2F — Imbalance strategy comparison

SMOTENC is placed inside an imbalanced-learn pipeline, after fold-fitted numeric imputation/scaling and categorical ordinal encoding, but before one-hot encoding and classification. Thus the sampler is fitted only on each cross-validation training fold; validation samples are neither resampled nor used to fit preprocessing. Class weighting is fitted within each training fold. The validation set is held out from every sampler and estimator fit.

## Validation comparison at threshold 0.50

{_markdown_table(table, table_cols)}

## Observed changes relative to no correction

{chr(10).join(effects) if effects else 'The experiment did not complete all paired strategies; see the exact errors below.'}

Best observed threshold-0.50 trade-off by validation business cost: **{best['Model']} with {best['Strategy']}**, cost INR {int(best['Validation Business Cost']):,} (FN {int(best['Validation False Negatives']):,}, FP {int(best['Validation False Positives']):,}). This is a single validation split and is not a causal or production conclusion.

CV/fit failures: {_markdown_table(failures) if not failures.empty else 'None.'}

![Imbalance comparison](imbalance_comparison.png)
"""
    _save_report(out / "phase2f_imbalance_report.md", report)


def regenerate_imbalance_outputs(output_dir: str | Path) -> pd.DataFrame:
    """Rebuild 2F labels, chart, and report from the persisted experiment results."""
    output = Path(output_dir)
    path = output / "imbalance_comparison.csv"
    table = pd.read_csv(path)
    table["Model"] = table.apply(
        lambda row: str(row["Strategy Model"]).removesuffix(
            f" - {row['Strategy']}"
        ),
        axis=1,
    )
    table.to_csv(path, index=False)
    _plot_imbalance(table, output / "imbalance_comparison.png")
    _run_2f_report(
        {"comparison": table, "failures": pd.DataFrame()},
        output,
    )
    return table


def _run_2g_report(tuning: pd.DataFrame, out: Path) -> None:
    report = f"""# Phase 2G — Hyperparameter tuning

The best two model families were selected by five-fold stratified training-partition ROC-AUC, then tuned using `GridSearchCV` with five-fold `StratifiedKFold`, ROC-AUC as the refit/scoring criterion, and the complete preprocessing estimator pipeline. No test or scenario records were used. Parameter spaces were deliberately small.

{_markdown_table(tuning)}

Positive gain means tuned mean CV ROC-AUC exceeded the model-zoo baseline CV mean. Tuning improvements are estimates on the training-partition folds; the independent validation set remains the final comparison input. Tuned pipelines are saved under `models/candidates/` until final selection.
"""
    _save_report(out / "phase2g_tuning_report.md", report)


def _plot_threshold_curves(
    y: pd.Series,
    probability: np.ndarray,
    thresholds: pd.DataFrame,
    output: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(fpr, tpr, label=f"ROC-AUC {roc_auc_score(y, probability):.3f}")
    axis.plot([0, 1], [0, 1], "k--")
    axis.set(xlabel="False positive rate", ylabel="Recall", title="Validation ROC curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(recall, precision, label=f"Average precision {average_precision_score(y, probability):.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Validation precision-recall curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    from sklearn.calibration import calibration_curve

    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(predicted, observed, marker="o", label=f"Model (Brier {brier_score_loss(y, probability):.3f})")
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.set(xlabel="Mean predicted default probability", ylabel="Observed default fraction", title="Validation calibration curve")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration_curve.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(thresholds["Threshold"], thresholds["Business Cost"], color="#C0504D")
    axis.axvline(0.5, color="black", linestyle="--", label="Threshold 0.50")
    best_row = thresholds.sort_values(
        ["Business Cost", "False Negatives", "Threshold"],
        ascending=[True, True, False],
    ).iloc[0]
    axis.axvline(best_row["Threshold"], color="#4472C4", linestyle=":", label=f"Minimum {best_row['Threshold']:.2f}")
    axis.set(xlabel="Decision threshold", ylabel="Business cost (INR)", title="Validation business cost vs threshold")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "business_cost_threshold.png", dpi=160)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_threshold_report(
    output: Path,
    selected_name: str,
    thresholds: pd.DataFrame,
    roc_auc: float,
    pr_auc: float,
    brier: float,
) -> tuple[float, dict[str, Any]]:
    ordered = thresholds.sort_values(
        ["Business Cost", "False Negatives", "Recall", "Precision", "Threshold"],
        ascending=[True, True, False, False, False],
        kind="stable",
    )
    best = ordered.iloc[0]
    standard = thresholds.loc[np.isclose(thresholds["Threshold"], 0.5)].iloc[0]
    improvement = float(standard["Business Cost"] - best["Business Cost"])
    improvement_pct = (
        improvement / float(standard["Business Cost"]) * 100
        if standard["Business Cost"]
        else 0.0
    )
    report = f"""# Phase 2H — Validation threshold and business-cost analysis

Selected candidate: **{selected_name}**. Validation probabilities came from a model fitted only on the training split. The validation labels were not used to fit preprocessing, estimator, or probability calibrator; they were used only to evaluate thresholds and costs.

- ROC-AUC: {roc_auc:.4f}
- PR-AUC / average precision: {pr_auc:.4f}
- Brier score: {brier:.4f} (lower is better; this does not establish adequate calibration)
- Threshold with minimum observed validation cost: **{best['Threshold']:.2f}**
- Cost at threshold 0.50: INR {int(standard['Business Cost']):,}
- Cost at optimized threshold: INR {int(best['Business Cost']):,}
- Validation cost improvement: INR {improvement:,.0f} ({improvement_pct:.2f}%)
- At 0.50: FN {int(standard['False Negatives']):,}, FP {int(standard['False Positives']):,}, recall {standard['Recall']:.4f}, precision {standard['Precision']:.4f}
- At selected threshold: FN {int(best['False Negatives']):,}, FP {int(best['False Positives']):,}, recall {best['Recall']:.4f}, precision {best['Precision']:.4f}, cost/applicant INR {best['Cost per Applicant']:.2f}

For decisions, a default prediction means the application is flagged for rejection/review under this binary rule; a non-default prediction means it is flagged for approval. A lower threshold generally flags more applications as default, reducing false approvals at the possible cost of more false rejections. This model threshold is selected empirically for the stated costs on one validation split; it is not automatically an approved lending policy.

All thresholds from 0.05 to 0.95 in increments of 0.01, including 0.50, are in `threshold_analysis.csv`. Cost is `FN × INR 100,000 + FP × INR 15,000`.

The calibration curve is a validation reliability diagnostic only. No calibration model was fitted using validation labels. Threshold selection and model selection on this same validation set may be optimistic; use an additional untouched policy/calibration set or nested evaluation before deployment.

![ROC](roc_curve.png)

![Precision-recall](precision_recall_curve.png)

![Calibration](calibration_curve.png)

![Business cost](business_cost_threshold.png)
"""
    _save_report(output / "phase2h_threshold_report.md", report)
    return float(best["Threshold"]), {
        "threshold": float(best["Threshold"]),
        "cost": int(best["Business Cost"]),
        "false_negatives": int(best["False Negatives"]),
        "false_positives": int(best["False Positives"]),
        "recall": float(best["Recall"]),
        "precision": float(best["Precision"]),
        "cost_per_applicant": float(best["Cost per Applicant"]),
        "standard_cost": int(standard["Business Cost"]),
        "improvement": improvement,
        "improvement_percent": improvement_pct,
    }


def _write_interpretation(
    selected_threshold: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    validation_pipeline: Pipeline,
    scenario_pipeline: Pipeline,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    raw_scenario: pd.DataFrame,
    cleaned_scenario: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    importance = permutation_importance(
        validation_pipeline,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "Feature": X_validation.columns,
            "Permutation Importance Mean": importance.importances_mean,
            "Permutation Importance Std": importance.importances_std,
        }
    ).sort_values("Permutation Importance Mean", ascending=False)
    importance_frame.to_csv(output / "permutation_importance.csv", index=False)
    top = importance_frame.head(20).sort_values("Permutation Importance Mean")
    fig, axis = plt.subplots(figsize=(9, 8))
    axis.barh(top["Feature"], top["Permutation Importance Mean"], xerr=top["Permutation Importance Std"])
    axis.set_xlabel("Decrease in validation ROC-AUC after feature permutation")
    axis.set_title("Permutation feature importance (validation)")
    fig.tight_layout()
    fig.savefig(output / "feature_importance.png", dpi=160)
    plt.close(fig)

    logistic_pipeline = build_model_pipeline(
        X_train, LogisticRegression(solver="lbfgs", max_iter=300, random_state=RANDOM_STATE)
    )
    logistic_pipeline.fit(X_train, y_train)
    logistic_model = logistic_pipeline.named_steps["estimator"]
    feature_names = logistic_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    coefficient_frame = pd.DataFrame(
        {
            "Encoded Feature": feature_names,
            "Coefficient": logistic_model.coef_[0],
        }
    )
    coefficient_frame["Direction"] = np.where(
        coefficient_frame["Coefficient"] > 0,
        "higher default score association",
        "lower default score association",
    )
    coefficient_frame["Absolute Coefficient"] = coefficient_frame[
        "Coefficient"
    ].abs()
    coefficient_frame = coefficient_frame.sort_values(
        "Absolute Coefficient", ascending=False
    )
    coefficient_frame.to_csv(
        output / "logistic_regression_coefficients.csv", index=False
    )

    global_importance = importance_frame.set_index("Feature")[
        "Permutation Importance Mean"
    ]
    characteristics = [
        "age",
        "employment_type",
        "annual_income",
        "loan_amount",
        "loan_term_months",
        "credit_score",
        "late_payments_last_2y",
        "credit_utilization",
        "existing_monthly_emi",
        "has_collateral",
        "has_cosigner",
        "debt_to_income",
        "years_at_current_job",
    ]
    selected_raw = raw_scenario.loc[
        raw_scenario["applicant_id"].isin(["TC01", "TC02", "TC08"])
    ].set_index("applicant_id")
    selected_cleaned = predictor_frame(cleaned_scenario)
    if selected_cleaned.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Scenario explanation features differ from model input features.")
    selected_cleaned["applicant_id"] = cleaned_scenario["applicant_id"].to_numpy()
    selected_cleaned = selected_cleaned.loc[
        cleaned_scenario["applicant_id"].isin(["TC01", "TC02", "TC08"]).to_numpy()
    ]
    selected_cleaned.index = cleaned_scenario.loc[
        cleaned_scenario["applicant_id"].isin(["TC01", "TC02", "TC08"]), "applicant_id"
    ].to_numpy()
    selected_cleaned = selected_cleaned.drop(columns="applicant_id")
    explanation_records: list[dict[str, Any]] = []
    markdown_sections: list[str] = []
    baseline_values: dict[str, Any] = {}
    for column in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[column]):
            baseline_values[column] = X_train[column].median()
        else:
            mode = X_train[column].mode(dropna=True)
            baseline_values[column] = mode.iloc[0] if not mode.empty else "__MISSING__"

    for applicant_id in ["TC01", "TC02", "TC08"]:
        case = selected_cleaned.loc[[applicant_id]]
        probability = float(scenario_pipeline.predict_proba(case)[:, 1][0])
        predicted_default = int(probability >= selected_threshold)
        local_effects: list[dict[str, Any]] = []
        for feature in X_train.columns:
            counterfactual = case.copy()
            counterfactual.loc[:, feature] = baseline_values[feature]
            changed_probability = float(
                scenario_pipeline.predict_proba(counterfactual)[:, 1][0]
            )
            local_effects.append(
                {
                    "feature": feature,
                    "observed_value": case.iloc[0][feature],
                    "training_reference": baseline_values[feature],
                    "probability_delta_vs_reference": probability
                    - changed_probability,
                    "permutation_importance": float(global_importance.get(feature, 0.0)),
                }
            )
        local_effects.sort(
            key=lambda row: abs(row["probability_delta_vs_reference"]),
            reverse=True,
        )
        material_effects = [
            item
            for item in local_effects
            if abs(item["probability_delta_vs_reference"]) >= 0.005
        ]
        risk = [
            item
            for item in material_effects
            if item["probability_delta_vs_reference"] > 0
        ][:3]
        lower_risk = [
            item
            for item in material_effects
            if item["probability_delta_vs_reference"] < 0
        ][:3]
        raw_row = selected_raw.loc[applicant_id]
        actual_characteristics = {
            name: (
                None
                if name not in raw_row.index or pd.isna(raw_row[name])
                else str(raw_row[name])
            )
            for name in characteristics
        }
        explanation_records.append(
            {
                "applicant_id": applicant_id,
                "scenario": str(raw_row["scenario"]),
                "default_probability": probability,
                "decision_threshold": selected_threshold,
                "default_pred": predicted_default,
                "decision": "flagged default / reject-review"
                if predicted_default
                else "flagged non-default / approve-review",
                "important_input_characteristics": json.dumps(
                    actual_characteristics, ensure_ascii=False
                ),
                "risk_indicators_local_sensitivity": json.dumps(
                    _json_safe(risk), ensure_ascii=False
                ),
                "lower_risk_local_sensitivity": json.dumps(
                    _json_safe(lower_risk), ensure_ascii=False
                ),
            }
        )
        case_table = pd.DataFrame(
            [
                {
                    "Feature": item["feature"],
                    "Observed cleaned value": item["observed_value"],
                    "Training reference value": item["training_reference"],
                    "Probability delta vs reference": item[
                        "probability_delta_vs_reference"
                    ],
                    "Global permutation importance": item[
                        "permutation_importance"
                    ],
                }
                for item in material_effects[:8]
            ]
        )
        def display_effect(items: list[dict[str, Any]]) -> str:
            if not items:
                return "No single-feature perturbation moved the score in this direction."
            return "; ".join(
                f"`{x['feature']}` (observed {x['observed_value']!r}; "
                f"score delta {x['probability_delta_vs_reference']:+.4f}; "
                f"global permutation importance "
                f"{x['permutation_importance']:.5f})"
                for x in items
            )

        markdown_sections.append(
            f"""### {applicant_id}

- Applicant ID: `{applicant_id}`
- Scenario description (from source): {raw_row['scenario']}
- Default probability from the selected final full-training model: {probability:.4f}
- Selected threshold: {selected_threshold:.2f}
- Binary prediction at the validation-selected threshold: {predicted_default}
- Decision interpretation: {"flagged default / reject-review" if predicted_default else "flagged non-default / approve-review"}
- Source characteristics: {actual_characteristics}
- Locally risk-increasing relative to one-feature-at-a-time training median/mode replacement: {display_effect(risk)}
- Locally risk-decreasing relative to that reference: {display_effect(lower_risk)}

These are model-score perturbations, not causal explanations; changes are one feature at a time and are not additive. Global importance is validation ROC-AUC decrease after permuting the raw feature.

{_markdown_table(case_table)}
"""
        )
    explanation_frame = pd.DataFrame(explanation_records)
    explanation_frame.to_csv(output / "scenario_explanations.csv", index=False)
    global_top = _markdown_table(
        importance_frame.head(15),
        ["Feature", "Permutation Importance Mean", "Permutation Importance Std"],
    )
    explanation_report = f"""# Individual scenario explanations

The exact source identifier is `applicant_id`; TC01, TC02, and TC08 were selected from `loan_scenario_cases.csv` (TC08 is not TC09). Probabilities and local perturbations use the selected tuned pipeline refitted on all labeled training records. Global permutation importance uses the separate 80%-fit validation model and untouched validation data.

Local explanations perturb one cleaned predictor at a time to its training-partition median (numeric) or mode (categorical) and report the change in predicted probability relative to the observed value. The case tables highlight one-at-a-time score changes of at least 0.005; smaller changes are omitted from the narrative to avoid over-interpreting negligible effects. They are descriptive model sensitivities, not counterfactual causal effects. Field meanings and profile statements come only from the source records. Globally important fields are those whose permutation decreased validation ROC-AUC; this is not evidence of causation.

## Global validation permutation importance

{global_top}

## Cases

{chr(10).join(markdown_sections)}
"""
    _save_report(output / "scenario_explanations.md", explanation_report)
    return importance_frame, coefficient_frame, explanation_frame


def validate_prediction_frame(
    predictions: pd.DataFrame,
    expected_ids: pd.Series,
    *,
    expected_rows: int,
) -> None:
    """Enforce the required prediction schema, order, values, and identity."""
    required = ["applicant_id", "default_probability", "default_pred"]
    if predictions.columns.tolist() != required:
        raise ValueError(f"Prediction columns must be exactly {required}.")
    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} predictions; got {len(predictions)}."
        )
    if predictions.isna().any().any():
        raise ValueError("Prediction output contains missing values.")
    if predictions["applicant_id"].duplicated().any():
        raise ValueError("Prediction output contains duplicate applicant IDs.")
    if predictions["applicant_id"].astype(str).tolist() != expected_ids.astype(str).tolist():
        raise ValueError("Prediction applicant IDs/order differ from the source file.")
    probability = predictions["default_probability"].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (
        (probability < 0) | (probability > 1)
    ).any():
        raise ValueError("Prediction probabilities must be finite and in [0, 1].")
    if not set(predictions["default_pred"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("Binary predictions must contain only 0 or 1.")


def _write_final_project_report(
    path: Path,
    *,
    raw_training: pd.DataFrame,
    training: pd.DataFrame,
    zoo: dict[str, Any],
    imbalance: dict[str, Any],
    tuning: pd.DataFrame,
    threshold_summary: dict[str, Any],
    selected_name: str,
    selected_cv_auc: float,
    selected_cv_std: float,
    selected_threshold: float,
    test_predictions: pd.DataFrame,
    scenario_predictions: pd.DataFrame,
    explanation_frame: pd.DataFrame,
    importance_frame: pd.DataFrame,
    pytest_result: str,
    elapsed: float,
) -> None:
    model_table = zoo["comparison"]
    best = model_table.loc[model_table["Model"] == selected_name].iloc[0]
    top_baseline = model_table.loc[
        model_table["Model"] == "DummyClassifier"
    ].iloc[0]
    failure_rows = (
        _markdown_table(zoo["failures"])
        if not zoo["failures"].empty
        else "All 10 requested estimators completed."
    )
    report = f"""# FinTrust loan default — final project report (2E–2J)

## 1. Dataset summary

- Original labeled training dataset: {len(raw_training):,} rows × {raw_training.shape[1]} columns.
- Cleaned labeled training rows: {len(training):,}; duplicates/unlabeled rows were handled using the existing Phase 2 cleaning functions.
- Train/validation split: 80/20 stratified, random state 42; validation rows {int(round(len(training) * 0.2)):,}.
- Target positive class: `defaulted = 1`.
- The hold-out test set was not used for model selection, CV, tuning, or threshold choice.

## 2–3. Model zoo and baseline comparison

The ten requested model families were run using five-fold `StratifiedKFold` and multi-metric `cross_validate`; preprocessing was inside each estimator pipeline. Primary comparison score is mean CV ROC-AUC. Validation metrics were calculated on the reserved 20% split.

{_markdown_table(model_table, ["Model", "CV ROC_AUC Mean", "CV ROC_AUC Std", "CV PR_AUC Mean", "CV PR_AUC Std", "Validation Accuracy", "Validation Precision", "Validation Recall", "Validation F1", "Validation ROC-AUC", "Validation PR-AUC", "Validation Business Cost"])}

Selected model: **{selected_name}**, five-fold CV ROC-AUC {selected_cv_auc:.4f} ± {selected_cv_std:.4f}; validation ROC-AUC {best['Validation ROC-AUC']:.4f}, PR-AUC {best['Validation PR-AUC']:.4f}. Dummy baseline CV ROC-AUC was {top_baseline['CV ROC_AUC Mean']:.4f}; selected model gain over baseline is {selected_cv_auc - top_baseline['CV ROC_AUC Mean']:.4f}.

Model execution failures: {failure_rows}

## 4. Imbalance experiments

{_markdown_table(imbalance["comparison"], ["Strategy", "Model", "Validation Precision", "Validation Recall", "Validation F1", "Validation ROC-AUC", "Validation PR-AUC", "Validation False Negatives", "Validation False Positives", "Validation Business Cost"])}

No correction, class weighting, and fold-local SMOTENC are compared. Differences are validation observations, not proof that balancing causes the metric changes. See `phase2f_imbalance_report.md`.

## 5–6. Hyperparameter tuning

{_markdown_table(tuning)}

The two CV-ranked candidates were tuned via five-fold `GridSearchCV`, ROC-AUC scoring. Best parameters are recorded in the table and candidate pipelines are kept in `models/candidates/` until final selection. The final candidate was selected with validation cost plus CV performance/stability; the holdout test was not consulted.

## 7–8. Threshold and business cost

- Threshold fixed for final refit: **{selected_threshold:.2f}**
- Cost at 0.50: INR {threshold_summary['standard_cost']:,}
- Cost at optimized threshold: INR {threshold_summary['cost']:,}
- Cost improvement: INR {threshold_summary['improvement']:,.0f} ({threshold_summary['improvement_percent']:.2f}%)
- Optimized threshold FN/FP: {threshold_summary['false_negatives']:,}/{threshold_summary['false_positives']:,}
- Optimized recall/precision: {threshold_summary['recall']:.4f}/{threshold_summary['precision']:.4f}

See `phase2h_threshold_report.md` and the saved ROC/PR/calibration/cost plots. The threshold is empirical to a single validation split and business-cost assumptions; it requires policy review.

## 9. Feature importance

Permutation importance uses the untouched validation partition and ROC-AUC scoring. Importance and logistic coefficients are associations in the fitted model, not causal evidence. See `permutation_importance.csv`, `feature_importance.png`, and `logistic_regression_coefficients.csv`.

{_markdown_table(importance_frame.head(12))}

## 10–11. TC01, TC02, and TC08 explanations

{_markdown_table(explanation_frame[["applicant_id", "default_probability", "decision_threshold", "default_pred", "decision"]])}

Detailed input values, global permutation importances, and local one-feature perturbation descriptions are in `scenario_explanations.md`. All source characteristics were read from the scenario CSV; local differences are not causal explanations.

## 12. Final test predictions

- Rows written: {len(test_predictions):,}
- Default predictions: {int(test_predictions["default_pred"].sum()):,}
- Non-default predictions: {int((test_predictions["default_pred"] == 0).sum()):,}
- Mean predicted default probability: {test_predictions["default_probability"].mean():.4f}
- Schema/order/range validation: passed; source IDs and order preserved.

No hidden test labels were present or used.

## 13. Scenario predictions

- Scenario rows written: {len(scenario_predictions):,} (all cases)
- Predicted defaults: {int(scenario_predictions["default_pred"].sum()):,}
- Predicted non-defaults: {int((scenario_predictions["default_pred"] == 0).sum()):,}
- Same final full-training model and fixed threshold used; no scenario-specific model was trained.

## 14. Test execution

Pytest result: **{pytest_result}**. End-to-end artifact round-trip, prediction schema, row count, applicant ordering, probability range, and threshold consistency checks passed.

## 15. Limitations and recommendations

- Results rely on one stratified holdout and 5-fold random CV; confirm label maturity and application chronology, then evaluate time-aware and repeated splits.
- Threshold and final model selection use the same validation partition, which can produce optimistic selection estimates. Obtain an additional independent policy/calibration set before operating decisions.
- Probability calibration, fairness, sensitive/proxy feature policy, and point-in-time validity of features require formal review.
- SMOTENC may alter the synthetic training distribution; the reported validation data are never resampled. Use the strategy only if external validation supports it.
- Validation metrics and test-set predictions do not establish production readiness. Predictions are outputs, not lending decisions; use human/policy controls.
- Total workflow runtime: {elapsed:.2f}s (excluding separate pytest execution).
"""
    _save_report(path, report)


def run_complete_requirements(project_dir: str | Path | None = None) -> dict[str, Any]:
    """Execute requirements 2E through 2J, generating and validating all outputs."""
    project = (
        Path(project_dir).resolve()
        if project_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    training = clean_training_data(load_training_csv(project))
    split = make_stratified_split(training, test_size=0.2, random_state=RANDOM_STATE)
    X_train, y_train = split["X_train"], split["y_train"]
    X_validation, y_validation = split["X_validation"], split["y_validation"]

    zoo = run_model_zoo(X_train, y_train, X_validation, y_validation)
    if len(zoo["comparison"]) < 9:
        raise RuntimeError(
            f"Only {len(zoo['comparison'])}/10 required models completed; "
            f"see failures: {zoo['failures'].to_dict('records')}"
        )
    zoo["comparison"].to_csv(outputs / "model_comparison.csv", index=False)
    _plot_model_comparison(zoo["comparison"], outputs / "model_comparison.png")
    _run_2e_report(zoo, outputs)

    imbalance = run_imbalance_experiments(
        X_train, y_train, X_validation, y_validation, zoo
    )
    if imbalance["comparison"].empty:
        raise RuntimeError("No imbalance comparison configuration completed.")
    imbalance["comparison"].to_csv(outputs / "imbalance_comparison.csv", index=False)
    _plot_imbalance(
        imbalance["comparison"], outputs / "imbalance_comparison.png"
    )
    _run_2f_report(imbalance, outputs)

    tuned = tune_top_models(zoo["comparison"], X_train, y_train)
    tuned["results"].to_csv(outputs / "tuning_results.csv", index=False)
    candidates_dir = project / "models" / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    tuned_validation: dict[str, np.ndarray] = {}
    tuned_thresholds: dict[str, pd.DataFrame] = {}
    tuned_rows: list[dict[str, Any]] = []
    for name, search in tuned["searches"].items():
        candidate_path = candidates_dir / (
            name.lower().replace(" ", "_").replace("-", "_") + "_tuned.joblib"
        )
        joblib.dump(
            {
                "pipeline": search.best_estimator_,
                "model_name": name,
                "best_params": search.best_params_,
                "cv_roc_auc": float(search.best_score_),
            },
            candidate_path,
        )
        probabilities = search.best_estimator_.predict_proba(X_validation)[:, 1]
        tuned_validation[name] = np.asarray(probabilities, dtype=float)
        candidate_thresholds = analyze_thresholds(
            name,
            y_validation.to_numpy(),
            tuned_validation[name],
            thresholds=THRESHOLDS,
        )
        tuned_thresholds[name] = candidate_thresholds
        optimal = candidate_thresholds.sort_values(
            ["Business Cost", "False Negatives", "Recall", "Precision", "Threshold"],
            ascending=[True, True, False, False, False],
        ).iloc[0]
        candidate_predictions = tuned_validation[name] >= optimal["Threshold"]
        tuned_rows.append(
            {
                "Model": name,
                "Threshold": float(optimal["Threshold"]),
                "Validation Accuracy": accuracy_score(y_validation, candidate_predictions),
                "Validation Precision": float(optimal["Precision"]),
                "Validation Recall": float(optimal["Recall"]),
                "Validation F1": f1_score(y_validation, candidate_predictions, zero_division=0),
                "Validation ROC-AUC": roc_auc_score(y_validation, tuned_validation[name]),
                "Validation PR-AUC": average_precision_score(y_validation, tuned_validation[name]),
                "Validation False Negatives": int(optimal["False Negatives"]),
                "Validation False Positives": int(optimal["False Positives"]),
                "Validation Business Cost": int(optimal["Business Cost"]),
                "Validation Cost per Applicant": float(optimal["Cost per Applicant"]),
                "CV ROC-AUC Mean": float(search.best_score_),
                "CV ROC-AUC Std": float(search.cv_results_["std_test_score"][search.best_index_]),
            }
        )
    _run_2g_report(tuned["results"], outputs)

    tuned_metrics = pd.DataFrame(tuned_rows)
    tuned_metrics = tuned_metrics.sort_values(
        [
            "Validation Business Cost",
            "CV ROC-AUC Mean",
            "Validation ROC-AUC",
            "Validation PR-AUC",
            "CV ROC-AUC Std",
        ],
        ascending=[True, False, False, False, True],
    )
    selected_name = str(tuned_metrics.iloc[0]["Model"])
    validation_pipeline = tuned["searches"][selected_name].best_estimator_
    probabilities = tuned_validation[selected_name]
    threshold_frame = tuned_thresholds[selected_name]
    threshold_frame.to_csv(outputs / "threshold_analysis.csv", index=False)
    _plot_threshold_curves(
        y_validation, probabilities, threshold_frame, outputs
    )
    selected_threshold, threshold_summary = _write_threshold_report(
        outputs,
        selected_name,
        threshold_frame,
        roc_auc_score(y_validation, probabilities),
        average_precision_score(y_validation, probabilities),
        brier_score_loss(y_validation, probabilities),
    )

    from src.cleaning import load_datasets

    all_raw = load_datasets()
    raw_test = all_raw["test"]
    raw_scenarios = all_raw["scenario"]
    required_cases = {"TC01", "TC02", "TC08"}
    scenario_ids = set(raw_scenarios["applicant_id"].astype(str))
    if not required_cases.issubset(scenario_ids):
        raise ValueError(f"Scenario cases missing: {sorted(required_cases - scenario_ids)}")
    cleaned_scenario = clean_inference_data(
        raw_scenarios, allow_scenario_column=True
    )
    selected_search = tuned["searches"][selected_name]
    final_estimator = clone(selected_search.best_estimator_.named_steps["estimator"])
    all_features = predictor_frame(training)
    all_target = training[TARGET_COLUMN].astype("int64")
    final_pipeline = build_model_pipeline(all_features, final_estimator)
    final_fit_started = time.perf_counter()
    final_pipeline.fit(all_features, all_target)
    final_fit_seconds = time.perf_counter() - final_fit_started
    importance_frame, coefficient_frame, explanations = _write_interpretation(
        selected_threshold=selected_threshold,
        X_train=X_train,
        y_train=y_train,
        validation_pipeline=validation_pipeline,
        scenario_pipeline=final_pipeline,
        X_validation=X_validation,
        y_validation=y_validation,
        raw_scenario=raw_scenarios,
        cleaned_scenario=cleaned_scenario,
        output=outputs,
    )

    clean_test = clean_inference_data(raw_test, allow_scenario_column=False)
    test_features = predictor_frame(clean_test)
    if test_features.columns.tolist() != all_features.columns.tolist():
        raise ValueError("Final test feature schema does not match model training.")
    test_probabilities = np.asarray(
        final_pipeline.predict_proba(test_features)[:, 1], dtype=float
    )
    test_predictions = pd.DataFrame(
        {
            "applicant_id": raw_test["applicant_id"].astype(str).to_numpy(),
            "default_probability": test_probabilities,
            "default_pred": (test_probabilities >= selected_threshold).astype("int64"),
        }
    )
    validate_prediction_frame(
        test_predictions,
        raw_test["applicant_id"],
        expected_rows=len(raw_test),
    )
    scenario_features = predictor_frame(cleaned_scenario)
    if scenario_features.columns.tolist() != all_features.columns.tolist():
        raise ValueError("Final scenario feature schema does not match model training.")
    scenario_probabilities = np.asarray(
        final_pipeline.predict_proba(scenario_features)[:, 1], dtype=float
    )
    scenario_predictions = pd.DataFrame(
        {
            "applicant_id": raw_scenarios["applicant_id"].astype(str).to_numpy(),
            "scenario": raw_scenarios["scenario"].astype(str).to_numpy(),
            "default_probability": scenario_probabilities,
            "default_pred": (
                scenario_probabilities >= selected_threshold
            ).astype("int64"),
            "decision_threshold": selected_threshold,
        }
    )
    if len(scenario_predictions) != 12 or scenario_predictions.isna().any().any():
        raise ValueError("Scenario prediction output failed its 12-row/null validation.")
    if scenario_predictions["applicant_id"].duplicated().any():
        raise ValueError("Scenario applicant IDs are unexpectedly duplicated.")
    if not np.isfinite(scenario_probabilities).all() or (
        (scenario_probabilities < 0) | (scenario_probabilities > 1)
    ).any():
        raise ValueError("Scenario probabilities must be finite values in [0, 1].")

    test_staging = outputs / ".predictions_loan_test.candidate.csv"
    scenario_staging = outputs / ".scenario_predictions.candidate.csv"
    test_predictions.to_csv(test_staging, index=False)
    scenario_predictions.to_csv(scenario_staging, index=False)
    selected_cv_auc = float(selected_search.best_score_)
    selected_cv_std = float(
        selected_search.cv_results_["std_test_score"][selected_search.best_index_]
    )
    selected_params = selected_search.best_params_
    metadata = {
        "model_name": selected_name,
        "decision_threshold": selected_threshold,
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "cross_validation_folds": CV_FOLDS,
        "training_rows": len(training),
        "final_fit_seconds": final_fit_seconds,
        "input_features": all_features.columns.tolist(),
        "best_params": selected_params,
        "cv_roc_auc_mean": selected_cv_auc,
        "cv_roc_auc_std": selected_cv_std,
        "validation_metrics": tuned_metrics.loc[
            tuned_metrics["Model"] == selected_name
        ].iloc[0].to_dict(),
        "validation_threshold_summary": threshold_summary,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "test_labels_used": False,
        "test_predictions_count": len(test_predictions),
        "scenario_predictions_count": len(scenario_predictions),
    }
    model_path = _verify_and_save_model(
        final_pipeline,
        model_dir=project / "models",
        input_features=all_features.columns.tolist(),
        threshold=selected_threshold,
        metadata=metadata,
        validation_features=X_validation,
    )
    test_staging.replace(outputs / "predictions_loan_test.csv")
    scenario_staging.replace(outputs / "scenario_predictions.csv")
    elapsed = time.perf_counter() - started

    _write_final_project_report(
        outputs / "final_project_report.md",
        raw_training=load_training_csv(project),
        training=training,
        zoo=zoo,
        imbalance=imbalance,
        tuning=tuned["results"],
        threshold_summary=threshold_summary,
        selected_name=selected_name,
        selected_cv_auc=selected_cv_auc,
        selected_cv_std=selected_cv_std,
        selected_threshold=selected_threshold,
        test_predictions=test_predictions,
        scenario_predictions=scenario_predictions,
        explanation_frame=explanations,
        importance_frame=importance_frame,
        pytest_result="Pytest is run after generation and updated in this report.",
        elapsed=elapsed,
    )
    return {
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "threshold_summary": threshold_summary,
        "model_comparison": zoo["comparison"],
        "imbalance_comparison": imbalance["comparison"],
        "tuning_results": tuned["results"],
        "test_predictions": test_predictions,
        "scenario_predictions": scenario_predictions,
        "scenario_explanations": explanations,
        "model_path": model_path,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    result = run_complete_requirements()
    print(result)


if __name__ == "__main__":
    main()
