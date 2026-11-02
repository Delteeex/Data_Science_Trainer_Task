"""Phase 3 model development, validation, cost tuning, and artifact export."""

from __future__ import annotations

import os
import sys
import time
import warnings
import importlib.util
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cleaning import (
    TARGET_COLUMN,
    clean_inference_data,
    clean_training_data,
    get_predictor_columns,
    validate_training_data,
)
from src.features import build_model_pipeline, predictor_frame, to_dense_array

RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 100_000
FALSE_POSITIVE_COST = 15_000
BREAK_EVEN_THRESHOLD = FALSE_POSITIVE_COST / (
    FALSE_NEGATIVE_COST + FALSE_POSITIVE_COST
)
THRESHOLD_GRID = np.unique(
    np.concatenate((np.arange(0.01, 1.0, 0.01), [0.5, BREAK_EVEN_THRESHOLD]))
)


def prepare_datasets(
    data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Keep the Phase 2 cleaning entry point available for all supplied files."""
    from src.cleaning import load_datasets

    raw = load_datasets(data_dir)
    training = clean_training_data(raw["train"])
    testing = clean_inference_data(raw["test"], allow_scenario_column=False)
    scenarios = clean_inference_data(raw["scenario"], allow_scenario_column=True)

    predictor_columns = get_predictor_columns(training)
    if set(predictor_columns) != set(get_predictor_columns(testing)):
        raise ValueError("Training and test predictor columns do not match.")
    if set(predictor_columns) != set(get_predictor_columns(scenarios)):
        raise ValueError("Training and scenario predictor columns do not match.")
    validate_training_data(training)
    return {"train": training, "test": testing, "scenario": scenarios}


def load_training_csv(project_dir: str | Path) -> pd.DataFrame:
    """Read only the labeled training CSV for model development."""
    project = Path(project_dir).resolve()
    data_dir = project.parent.parent / "DS_Trainer_Task2_Data"
    training_path = data_dir / "loan_train.csv"
    if not training_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {training_path}")
    return pd.read_csv(training_path, low_memory=False)


def make_stratified_split(
    training: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Split normalized training rows; no preprocessing is fit before the split."""
    from sklearn.model_selection import train_test_split

    validate_training_data(training)
    predictors = get_predictor_columns(training)
    X = training.loc[:, predictors]
    y = training[TARGET_COLUMN].astype("int64")
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "y_train": y_train,
        "y_validation": y_validation,
    }


def save_cleaned_outputs(
    datasets: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save normalized intermediate CSVs without changing source files."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filenames = {
        "train": "loan_train_cleaned.csv",
        "test": "loan_test_cleaned.csv",
        "scenario": "loan_scenario_cases_cleaned.csv",
    }
    paths: dict[str, Path] = {}
    for name, filename in filenames.items():
        if name not in datasets:
            raise ValueError(f"Cleaned dataset bundle is missing {name!r}.")
        path = output / filename
        datasets[name].to_csv(path, index=False)
        paths[name] = path
    return paths


def candidate_estimators() -> dict[str, Any]:
    """Build efficient, seeded baselines using installed sklearn estimators."""
    hist_gradient = Pipeline(
        steps=[
            ("to_dense", FunctionTransformer(to_dense_array)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.08,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": hist_gradient,
    }


def analyze_thresholds(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray = THRESHOLD_GRID,
) -> pd.DataFrame:
    """Calculate confusion and business-cost metrics for each threshold."""
    y = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(probabilities) or not len(y):
        raise ValueError("Targets and probabilities must have the same nonzero length.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Threshold analysis requires binary targets encoded as 0/1.")

    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        cost = int(fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST)
        records.append(
            {
                "Model": model_name,
                "Threshold": float(threshold),
                "False Negatives": int(fn),
                "False Positives": int(fp),
                "Business Cost": cost,
                "Cost per Applicant": cost / len(y),
                "Recall": recall_score(y, predicted, zero_division=0),
                "Precision": precision_score(y, predicted, zero_division=0),
                "Specificity": tn / (tn + fp) if tn + fp else 0.0,
                "True Negatives": int(tn),
                "True Positives": int(tp),
                "Predicted Defaults": int(tp + fp),
                "Predicted Non-defaults": int(tn + fn),
                "Default Detection Rate": tp / (tp + fn) if tp + fn else 0.0,
                "False Approvals": int(fn),
                "False Rejections": int(fp),
            }
        )
    return pd.DataFrame(records)


def _metric_record(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=np.int64)
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y, predicted),
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1": f1_score(y, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities),
        "False Negatives": int(fn),
        "False Positives": int(fp),
        "True Negatives": int(tn),
        "True Positives": int(tp),
        "Specificity": tn / (tn + fp) if tn + fp else 0.0,
        "Business Cost": int(fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST),
        "Cost per Applicant": (
            fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST
        )
        / len(y),
        "Predicted Defaults": int(tp + fp),
        "Predicted Non-defaults": int(tn + fn),
        "Default Detection Rate": tp / (tp + fn) if tp + fn else 0.0,
    }


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    *,
    estimators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit each pipeline on training rows and evaluate only on validation rows."""
    estimator_map = estimators or candidate_estimators()
    fitted: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    threshold_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, Any]] = []

    for name, estimator in estimator_map.items():
        pipeline = build_model_pipeline(X_train, estimator)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        cv_started = time.perf_counter()
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            n_jobs=1,
            error_score="raise",
        )
        cv_seconds = time.perf_counter() - cv_started
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        validation_probs = np.asarray(
            pipeline.predict_proba(X_validation)[:, 1], dtype=np.float64
        )
        training_probs = np.asarray(pipeline.predict_proba(X_train)[:, 1], dtype=np.float64)
        if not np.isfinite(validation_probs).all() or (
            (validation_probs < 0) | (validation_probs > 1)
        ).any():
            raise ValueError(f"{name} produced invalid validation probabilities.")

        threshold_frame = analyze_thresholds(
            name, y_validation.to_numpy(), validation_probs
        )
        best_threshold_row = threshold_frame.sort_values(
            ["Business Cost", "False Negatives", "Recall", "Precision", "Threshold"],
            ascending=[True, True, False, False, False],
            kind="stable",
        ).iloc[0]
        threshold = float(best_threshold_row["Threshold"])
        validation_metrics = _metric_record(
            y_validation.to_numpy(), validation_probs, threshold
        )
        training_metrics = _metric_record(y_train.to_numpy(), training_probs, threshold)
        threshold_frame["Fit Seconds"] = fit_seconds
        threshold_frames.append(threshold_frame)
        comparison_rows.append(
            {
                "Model": name,
                "Threshold": threshold,
                **validation_metrics,
                "Training ROC-AUC": training_metrics["ROC-AUC"],
                "Training PR-AUC": training_metrics["PR-AUC"],
                "ROC-AUC Gap": training_metrics["ROC-AUC"]
                - validation_metrics["ROC-AUC"],
                "PR-AUC Gap": training_metrics["PR-AUC"]
                - validation_metrics["PR-AUC"],
                "CV ROC-AUC Mean": float(np.mean(cv_scores["test_roc_auc"])),
                "CV ROC-AUC Std": float(np.std(cv_scores["test_roc_auc"])),
                "CV PR-AUC Mean": float(np.mean(cv_scores["test_pr_auc"])),
                "CV PR-AUC Std": float(np.std(cv_scores["test_pr_auc"])),
                "CV Seconds": cv_seconds,
                "Fit Seconds": fit_seconds,
                "Total Model Seconds": cv_seconds + fit_seconds,
                "Convergence Warnings": sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                ),
            }
        )
        fitted[name] = pipeline
        validation_probabilities[name] = validation_probs
        print(
            f"{name}: fitted in {fit_seconds:.2f}s; "
            f"3-fold CV ROC-AUC={comparison_rows[-1]['CV ROC-AUC Mean']:.4f}"
            f"+/-{comparison_rows[-1]['CV ROC-AUC Std']:.4f}; "
            f"validation ROC-AUC={validation_metrics['ROC-AUC']:.4f}, "
            f"PR-AUC={validation_metrics['PR-AUC']:.4f}, "
            f"minimum cost=INR {validation_metrics['Business Cost']:,}"
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        [
            "Business Cost",
            "ROC-AUC",
            "PR-AUC",
            "Recall",
            "Precision",
            "CV ROC-AUC Std",
            "ROC-AUC Gap",
        ],
        ascending=[True, False, False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True)
    return {
        "comparison": comparison,
        "threshold_analysis": thresholds,
        "fitted_models": fitted,
        "validation_probabilities": validation_probabilities,
    }


def _plot_results(
    output_dir: Path,
    y_validation: pd.Series,
    comparison: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    validation_probabilities: dict[str, np.ndarray],
    selected_model: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "roc": output_dir / "roc_curves.png",
        "pr": output_dir / "precision_recall_curves.png",
        "confusion": output_dir / "selected_confusion_matrix.png",
        "cost": output_dir / "business_cost_threshold.png",
        "comparison": output_dir / "model_comparison.png",
    }
    labels = y_validation.to_numpy()

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, probabilities in validation_probabilities.items():
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        axis.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Validation ROC Curves")
    axis.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(paths["roc"], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for name, probabilities in validation_probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ap = average_precision_score(labels, probabilities)
        axis.plot(recall, precision, label=f"{name} (AP {ap:.3f})")
    axis.set(xlabel="Recall", ylabel="Precision", title="Validation Precision-Recall Curves")
    axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(paths["pr"], dpi=160)
    plt.close(fig)

    selected_row = comparison.loc[comparison["Model"] == selected_model].iloc[0]
    selected_threshold = float(selected_row["Threshold"])
    selected_probs = validation_probabilities[selected_model]
    predicted = selected_probs >= selected_threshold
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Applicants")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Non-default", "Default"],
        yticklabels=["Non-default", "Default"],
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"{selected_model} confusion matrix (threshold {selected_threshold:.4f})",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(paths["confusion"], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    for name, group in threshold_analysis.groupby("Model", sort=False):
        ordered = group.sort_values("Threshold")
        axis.plot(
            ordered["Threshold"],
            ordered["Business Cost"],
            label=name,
            linewidth=1.5,
        )
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default 0.50")
    axis.axvline(
        BREAK_EVEN_THRESHOLD,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Cost ratio {BREAK_EVEN_THRESHOLD:.3f}",
    )
    axis.set(
        xlabel="Decision threshold",
        ylabel="Validation business cost (₹)",
        title="Validation Business Cost by Decision Threshold",
    )
    axis.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(paths["cost"], dpi=160)
    plt.close(fig)

    chart_data = comparison.sort_values("Business Cost", ascending=True)
    fig, cost_axis = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(chart_data))
    cost_axis.bar(positions, chart_data["Business Cost"], color="#4472C4", alpha=0.8)
    cost_axis.set_ylabel("Minimum validation business cost (₹)")
    cost_axis.set_xticks(positions, chart_data["Model"], rotation=20, ha="right")
    auc_axis = cost_axis.twinx()
    auc_axis.plot(
        positions,
        chart_data["ROC-AUC"],
        color="#ED7D31",
        marker="o",
        linewidth=2,
        label="ROC-AUC",
    )
    auc_axis.set_ylabel("ROC-AUC")
    cost_axis.set_title("Validation Model Comparison at Each Model's Minimum-Cost Threshold")
    fig.tight_layout()
    fig.savefig(paths["comparison"], dpi=160)
    plt.close(fig)
    return paths


def _format_table(data: pd.DataFrame, columns: list[str]) -> str:
    display = data.loc[:, columns].copy()
    for column in display.select_dtypes(include="number").columns:
        if column in {"Business Cost", "False Negatives", "False Positives"}:
            display[column] = display[column].map(lambda value: f"{value:,.0f}")
        elif column == "Threshold":
            display[column] = display[column].map(lambda value: f"{value:.4f}")
        elif column.endswith("Seconds"):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
        else:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _write_evaluation_report(
    path: Path,
    comparison: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
    validation_probabilities: dict[str, np.ndarray],
    y_validation: pd.Series,
    *,
    train_rows: int,
    validation_rows: int,
    split_seconds: float,
    total_seconds: float,
    available_optional_models: list[str],
    chart_paths: dict[str, Path],
) -> dict[str, Any]:
    selected = comparison.iloc[0]
    selected_name = str(selected["Model"])
    selected_threshold_row = threshold_analysis.loc[
        (threshold_analysis["Model"] == selected_name)
        & np.isclose(threshold_analysis["Threshold"], selected["Threshold"])
    ].iloc[0]
    probabilities = validation_probabilities[selected_name]
    predictions = probabilities >= float(selected["Threshold"])
    matrix = confusion_matrix(y_validation, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    base_row = threshold_analysis.loc[
        (threshold_analysis["Model"] == selected_name)
        & np.isclose(threshold_analysis["Threshold"], 0.5)
    ].iloc[0]
    all_times = _format_table(
        comparison, ["Model", "CV Seconds", "Fit Seconds", "Total Model Seconds"]
    )
    skipped = (
        ", ".join(available_optional_models)
        if available_optional_models
        else "XGBoost, LightGBM, and CatBoost (not installed; intentionally not installed)"
    )
    report = f"""# FinTrust Loan Default Prediction — Phase 3 Evaluation

## Scope and reproducibility

This evaluation uses only labeled records from `loan_train.csv`. The original training file has 32,600 rows; Phase 2 cleaning removes exact duplicates and unlabeled records, leaving {train_rows:,} labeled records. An 80/20 stratified split with `random_state=42` produced {train_rows - validation_rows:,} model-training rows and {validation_rows:,} validation rows. The validation partition was not used for fitting any imputer, encoder, scaler, or classifier. The hold-out `loan_test.csv` and scenario cases were not loaded or used by the Phase 3 training workflow.

Preprocessing reuses `src/features.py` and is fitted inside each sklearn pipeline on the training partition only. Phase 2 target/leakage exclusions remain in effect. Positive class `1` means default. Each threshold is evaluated on the same held-out validation probabilities; the model selection and threshold results are therefore validation estimates and must not be interpreted as production performance.

Runtime details: split {split_seconds:.2f}s; total Phase 3 run {total_seconds:.2f}s. Per-model timing (three-fold cross-validation plus final training-partition fit):

{all_times}

False-negative cost = ₹{FALSE_NEGATIVE_COST:,}; false-positive cost = ₹{FALSE_POSITIVE_COST:,}. Optional estimators not evaluated: {skipped}.

## Model comparison

The table reports validation metrics at each model's threshold minimizing the specified validation cost. Selection prioritizes minimum validation business cost, then validation ROC-AUC, PR-AUC, recall, precision, lower three-fold ROC-AUC variability, and a smaller train-to-validation ROC-AUC gap.

{_format_table(comparison, ["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "CV ROC-AUC Mean", "CV ROC-AUC Std", "CV PR-AUC Mean", "CV PR-AUC Std", "Specificity", "False Negatives", "False Positives", "Business Cost", "Cost per Applicant", "CV Seconds", "Fit Seconds", "ROC-AUC Gap"])}

### Selected model and threshold

- **Selected model:** {selected_name}
- **Optimal validation threshold:** {float(selected["Threshold"]):.4f}
- **Accuracy:** {float(selected["Accuracy"]):.4f}
- **Precision:** {float(selected["Precision"]):.4f}
- **Default recall:** {float(selected["Recall"]):.4f}
- **F1:** {float(selected["F1"]):.4f}
- **ROC-AUC:** {float(selected["ROC-AUC"]):.4f}
- **PR-AUC / average precision:** {float(selected["PR-AUC"]):.4f}
- **Specificity:** {float(selected["Specificity"]):.4f}
- **Validation business cost:** ₹{int(selected["Business Cost"]):,}
- **Cost per validation applicant:** ₹{float(selected["Cost per Applicant"]):,.2f}
- **False negatives / false approvals:** {fn:,}
- **False positives / false rejections:** {fp:,}
- **Predicted defaults / non-defaults:** {int(selected["Predicted Defaults"]):,} / {int(selected["Predicted Non-defaults"]):,}
- **Default detection rate:** {float(selected["Default Detection Rate"]):.4f}

Confusion matrix, rows actual `[non-default, default]`, columns predicted `[non-default, default]`:

| | Predicted non-default | Predicted default |
|---|---:|---:|
| Actual non-default | {tn:,} | {fp:,} |
| Actual default | {fn:,} | {tp:,} |

At threshold 0.50, the selected model has {int(base_row["False Negatives"]):,} false negatives, {int(base_row["False Positives"]):,} false positives, and validation cost ₹{int(base_row["Business Cost"]):,}. The tuned threshold is selected from a 0.01–0.99 grid, including exactly 0.50 and the cost-only break-even reference ({BREAK_EVEN_THRESHOLD:.4f}); it is not assumed to be 0.50.

## Threshold analysis

`outputs/threshold_analysis.csv` contains the complete per-model sweep and the false-negative/false-positive counts, total cost, cost per applicant, recall, precision, specificity, predicted class counts, and default detection rate for each threshold. The total cost is `FN × ₹100,000 + FP × ₹15,000`. The mathematical break-even probability of 13.04% assumes well-calibrated probabilities and only these two costs; it is a reference point, not an operating-policy recommendation.

## Visualizations

![Validation ROC curves](roc_curves.png)

![Validation precision-recall curves](precision_recall_curves.png)

![Selected model confusion matrix](selected_confusion_matrix.png)

![Business cost by threshold](business_cost_threshold.png)

![Model comparison](model_comparison.png)

## Limitations and recommendation

- A single random stratified split does not establish model stability. `ROC-AUC Gap` and `PR-AUC Gap` compare training with validation on this split only; use repeated or time-aware validation after label maturity and application chronology are confirmed.
- Three-fold stratified cross-validation is also run inside the training partition with the complete preprocessing pipeline refitted independently in each fold. Its ROC-AUC/PR-AUC means and standard deviations provide a training-only stability indicator, but do not replace repeated/time-aware validation or an independent test evaluation.
- Threshold optimization on the same validation partition used for model selection can be optimistic. A separate calibration/threshold-selection partition or nested/repeated validation is needed before policy use.
- Class-weighted estimators can produce probabilities that need calibration. Thresholds optimized here are tied to the fitted model and validation population; review calibration and cost sensitivity.
- The feature set retains the Phase 2 exclusions. Point-in-time availability of debt-to-income and branch-code policy/fairness risks remain open questions in the Phase 2 report.
- Validation metrics and costs do not establish production readiness. No test-set or scenario predictions were produced.

**Recommendation:** retain {selected_name} as the Phase 3 validation candidate at threshold {float(selected["Threshold"]):.4f}, subject to calibration, repeated/time-aware validation, fairness review, and confirmation of feature timing and cost assumptions. Do not treat this single-split result as an approved production model.

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return {
        "model": selected_name,
        "threshold": float(selected["Threshold"]),
        "confusion_matrix": matrix.tolist(),
        "metrics": selected.to_dict(),
    }


def _verify_and_save_model(
    pipeline: Pipeline,
    *,
    model_dir: Path,
    input_features: list[str],
    threshold: float,
    metadata: dict[str, Any],
    validation_features: pd.DataFrame,
) -> Path:
    """Verify a temporary round-trip artifact before replacing any old model."""
    model_dir.mkdir(parents=True, exist_ok=True)
    final_path = model_dir / "final_model.joblib"
    candidate_path = model_dir / ".final_model.candidate.joblib"
    artifact = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "metadata": metadata,
        "input_features": input_features,
    }
    try:
        joblib.dump(artifact, candidate_path)
        loaded = joblib.load(candidate_path)
        if loaded["input_features"] != input_features:
            raise ValueError("Serialized model input feature list does not match training.")
        if not np.isclose(loaded["threshold"], threshold):
            raise ValueError("Serialized model threshold does not match validation selection.")
        if not isinstance(loaded["pipeline"], Pipeline):
            raise TypeError("Serialized model does not contain an sklearn Pipeline.")
        probabilities = loaded["pipeline"].predict_proba(validation_features)[:, 1]
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise ValueError("Serialized model produced invalid validation probabilities.")
        if list(loaded["pipeline"].feature_names_in_) != input_features:
            raise ValueError("Serialized model feature schema differs from training.")

        os.replace(candidate_path, final_path)
        verified = joblib.load(final_path)
        if not np.isclose(verified["threshold"], threshold):
            raise ValueError("Final model artifact failed threshold verification.")
        final_probabilities = verified["pipeline"].predict_proba(
            validation_features
        )[:, 1]
        if not np.isfinite(final_probabilities).all() or (
            (final_probabilities < 0) | (final_probabilities > 1)
        ).any():
            raise ValueError("Final model artifact failed probability verification.")
        return final_path
    finally:
        if candidate_path.exists():
            candidate_path.unlink()


def run_phase3(project_dir: str | Path | None = None) -> dict[str, Any]:
    """Train baselines, select validation cost threshold, and save verified outputs."""
    started = time.perf_counter()
    project = (
        Path(project_dir).resolve()
        if project_dir is not None
        else Path(__file__).resolve().parents[1]
    )
    raw_train = load_training_csv(project)
    training = clean_training_data(raw_train)
    validate_training_data(training)
    prepared_at = time.perf_counter()
    split = make_stratified_split(training, test_size=0.2, random_state=RANDOM_STATE)
    split_seconds = time.perf_counter() - prepared_at

    print(
        f"Training {len(split['y_train']):,}; validation {len(split['y_validation']):,}; "
        f"default rate train={split['y_train'].mean():.4%}, "
        f"validation={split['y_validation'].mean():.4%}"
    )
    evaluated = evaluate_candidates(
        split["X_train"],
        split["y_train"],
        split["X_validation"],
        split["y_validation"],
    )
    comparison = evaluated["comparison"]
    thresholds = evaluated["threshold_analysis"]
    selected_name = str(comparison.iloc[0]["Model"])
    selected_pipeline = evaluated["fitted_models"][selected_name]
    selected_threshold = float(comparison.iloc[0]["Threshold"])

    outputs = project / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(outputs / "model_comparison.csv", index=False)
    thresholds.to_csv(outputs / "threshold_analysis.csv", index=False)
    chart_paths = _plot_results(
        outputs,
        split["y_validation"],
        comparison,
        thresholds,
        evaluated["validation_probabilities"],
        selected_name,
    )
    optional_models = [
        name
        for name, package in (
            ("XGBoost", "xgboost"),
            ("LightGBM", "lightgbm"),
            ("CatBoost", "catboost"),
        )
        if importlib.util.find_spec(package) is not None
    ]
    selected_row = comparison.iloc[0]
    input_features = split["X_train"].columns.tolist()
    artifact_metadata = {
        "model_name": selected_name,
        "decision_threshold": selected_threshold,
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "training_rows": len(split["y_train"]),
        "validation_rows": len(split["y_validation"]),
        "input_features": input_features,
        "validation_metrics": selected_row.to_dict(),
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "validation_only_threshold": True,
    }
    model_path = _verify_and_save_model(
        selected_pipeline,
        model_dir=project / "models",
        input_features=input_features,
        threshold=selected_threshold,
        metadata=artifact_metadata,
        validation_features=split["X_validation"],
    )
    elapsed = time.perf_counter() - started
    report_result = _write_evaluation_report(
        outputs / "phase3_model_evaluation.md",
        comparison,
        thresholds,
        evaluated["validation_probabilities"],
        split["y_validation"],
        train_rows=len(training),
        validation_rows=len(split["y_validation"]),
        split_seconds=split_seconds,
        total_seconds=elapsed,
        available_optional_models=optional_models,
        chart_paths=chart_paths,
    )
    return {
        **report_result,
        "comparison": comparison,
        "threshold_analysis": thresholds,
        "model_path": model_path,
        "elapsed_seconds": elapsed,
        "chart_paths": chart_paths,
    }


def main() -> None:
    from src.phase2e_2j import run_complete_requirements

    result = run_complete_requirements()
    print(f"Selected model: {result['selected_model']}")
    print(f"Threshold: {result['threshold']:.2f}")
    print(f"Saved final model: {result['model_path']}")
    print(f"Test rows: {len(result['test_predictions']):,}")
    print(f"Scenario rows: {len(result['scenario_predictions']):,}")
    print(f"Total elapsed: {result['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    main()
