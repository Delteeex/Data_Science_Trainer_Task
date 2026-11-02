# FinTrust Loan Default Prediction (Task 2)

This project predicts 12-month loan default risk from application data. The saved pipeline includes cleaning, preprocessing, HistGradientBoosting, and a validation-selected decision threshold of **0.11**. Predictions support review; they are not automatic lending decisions.

## Project layout

- `src/` — cleaning, feature processing, training and 2E–2J workflow.
- `notebooks/task2_modelling.ipynb` — workflow and review of all 12 scenario results.
- `tests/` — unit and saved-output integration tests.
- `models/final_model.joblib` — final pipeline, threshold, and metadata.
- `outputs/` — reports, charts, model results, and generated predictions.
- `predict.py` — command-line inference for raw CSV input.

The original input files are expected at `../../DS_Trainer_Task2_Data/` relative to this project directory:
`loan_train.csv`, `loan_test.csv`, and `loan_scenario_cases.csv`.

## Environment

Use Python 3.11 or a compatible version with pandas, NumPy, matplotlib, seaborn, scikit-learn, SciPy, joblib, pytest, and imbalanced-learn installed. From the project directory, install the required packages if they are not already available:

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib pytest imbalanced-learn
```

## Run and evaluate the existing project

Open `notebooks/task2_modelling.ipynb` in VS Code/Jupyter. Its workflow cell runs the complete 2E–2J workflow, including model comparison, imbalance experiments, tuning, threshold analysis, interpretation, final refit, and predictions. This is a long-running experiment and overwrites generated outputs when rerun; reviewing the existing results does not require rerunning it.

To run the same workflow from the project directory:

```powershell
python src\train.py
```

Run the test suite:

```powershell
python -m pytest -q
```

Review these outputs to evaluate the persisted experiment:

- `outputs\final_project_report.md` — consolidated results and limitations.
- `outputs\model_comparison.csv` — ten models, validation metrics, and five-fold CV.
- `outputs\imbalance_comparison.csv` — no correction, class weighting, and SMOTENC.
- `outputs\tuning_results.csv` — tuning results for the two leading candidates.
- `outputs\threshold_analysis.csv` — threshold sweep and business costs.
- `outputs\scenario_predictions.csv` and `outputs\scenario_explanations.md` — all 12 scenario predictions and case review.
- `outputs\predictions_loan_test.csv` — predictions for the 8,000 unlabeled test rows.
- `outputs\roc_curves.png`, `outputs\precision_recall_curves.png`, `outputs\calibration_curve.png`, and `outputs\business_cost_threshold.png` — evaluation plots.

Test-set labels are not available in this project, so test predictions cannot be scored for accuracy, recall, or business cost. Validation results are estimates and do not establish production readiness.

## Predict from a raw CSV with the CLI

From the project directory, pass an input CSV and an output path:

```powershell
python predict.py --input any.csv --output preds.csv
```

The input must contain `applicant_id` and the application fields represented by the supplied datasets. Dirty numeric/category formats and unseen categories are passed through the shared cleaning and saved preprocessing pipeline. The CLI loads `models\final_model.joblib`, uses its stored 0.11 threshold, and writes `applicant_id`, `default_probability`, and `default_pred`. Missing IDs or a feature-schema mismatch produce an error.

For example, process all 12 supplied scenario rows to a separate CLI output:

```powershell
python predict.py --input ..\..\DS_Trainer_Task2_Data\loan_scenario_cases.csv --output outputs\scenario_cli_predictions.csv
```

The already-generated deliverable is `outputs\scenario_predictions.csv`; the example above creates an additional inference output and is not required for normal review.

## Scenario interpretation and cautions

The notebook reviews each scenario against its profile. TC01 and TC02 behave consistently with their low- and high-risk descriptions; TC08/TC09 show a lower score with collateral/co-signer, a directional sanity check rather than a causal effect. TC03 and TC10 need review because of missing data. TC11 has a negative loan amount and should be corrected or rejected as invalid input—the model score is not an approval recommendation. Unseen categories are accepted by the pipeline, but their predictions are not evidence that the category is safe.

The threshold 0.11 reduces false negatives at the cost of more false positives under the stated error costs. Confirm target timing, feature availability, fairness, calibration, and the threshold with business owners before any operational use.
