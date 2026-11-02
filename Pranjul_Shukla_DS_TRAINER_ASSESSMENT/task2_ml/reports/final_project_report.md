# FinTrust Loan Default Prediction — Final Project Report (Requirements 2E–2J)

## 1. Project introduction and business objective

FinTrust needs to estimate whether a loan applicant will default, defined by the project brief as becoming 90+ days past due within the target horizon. The model outputs a default probability and a threshold-based flag; it is a decision-support signal, not an approved lending policy.

The stated error costs are asymmetric:

- False negative (a default is predicted as non-default): **₹1,00,000**.
- False positive (a non-default is predicted as default): **₹15,000**.

The working threshold was therefore selected by minimizing validation cost, not by assuming 0.50 is optimal. The source brief describes the horizon relative to disbursal, while the initial task wording described it relative to application. The business owner must confirm the authoritative anchor and label maturity policy before operational use.

## 2. Dataset overview

The source datasets are the supplied `DS_Trainer_Task2_Data\loan_train.csv`, `loan_test.csv`, and `loan_scenario_cases.csv`. The source files were not edited.

| Dataset | Original dimensions | Cleaned dimensions | Role |
|---|---:|---:|---|
| `loan_train.csv` | 32,600 × 35 | 31,650 × 37 | Historical labeled training data |
| `loan_test.csv` | 8,000 × 34 | 8,000 × 36 | Unlabeled hold-out prediction data |
| `loan_scenario_cases.csv` | 12 × 35 | 12 × 37 | Dirty/manual scenario profiles |

`defaulted` is present only in training. The raw training target has 32,239 labeled records and 361 missing labels. Among labeled raw records, 6,007 (18.63%) are defaults and 26,232 (81.37%) are non-defaults, a 4.37:1 non-default/default ratio. The raw target uses multiple text encodings; cleaning normalizes recognized values to binary 0/1. After duplicate and missing-label handling, 31,650 labeled training rows remain.

The feature schema is shared by the operational datasets, but raw dtypes and category vocabularies vary; the hand-written scenarios include inconsistent representations and unseen values. `application_date` is converted to application year/month. The cleaned files add `application_year`, `application_month`, and `credit_score_no_history` (the date replacement accounts for the net column count).

## 3. Data cleaning and preprocessing

Implemented in `src/cleaning.py` and `src/features.py`, with details in [phase2_cleaning_report.md](./phase2_cleaning_report.md):

- Exact duplicate training rows are removed (600 in the raw input); records with missing target labels are excluded from supervised fitting (350 additional rows after duplicate handling). Test and scenario rows are retained.
- Target encodings, category aliases, whitespace/case, booleans, numeric text, currency/unit forms, term formats, and supported date formats are normalized.
- Invalid/unparseable numerical values and infinities are treated as missing for pipeline imputation. Suspicious extremes are retained rather than blindly dropped.
- Numeric features use training-fold median imputation with missing indicators and scaling. Categorical features use constant imputation and one-hot encoding with unknown values ignored. The fitted transformer aligns the validation, test, and scenario inputs.
- Preprocessing is inside the sklearn pipelines and fitted only on the relevant training fold/split; validation, test, and scenarios do not fit preprocessing.

The feature selector excludes the target and the following fields:

| Excluded field | Reason recorded in the cleaning report |
|---|---|
| `applicant_id` | Identifier for linkage/output, not a predictor |
| `recovery_calls_made` | Post-application recovery activity; direct temporal leakage |
| `record_created_by` | Operational metadata with proxy/leakage risk |
| `country` | Constant in the supplied datasets |
| `loan_amount_lakhs` | Redundant alternate-unit principal field with inconsistent scenario representations |
| `interest_rate` | Application-time availability not confirmed |
| `scenario` | Hand-authored narrative, not an application-time feature |
| `defaulted` | Training label, never an input feature |

`branch_code` remains a model input. Its deployment availability, stability, and proxy/fairness implications require review. Point-in-time provenance for `debt_to_income` also needs confirmation.

## 4. Model Zoo comparison

Ten classifiers were evaluated using the cleaned training data: DummyClassifier, Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, AdaBoost, and calibrated linear Support Vector Machine. An 80/20 stratified split (`random_state=42`) produced 25,320 fit rows and 6,330 validation rows. Five-fold `StratifiedKFold` CV was run on the training partition with preprocessing in the model pipeline. The table reports validation classification metrics at the default threshold of 0.50; CV columns are mean ± standard deviation. “Mean fold fit” is the persisted mean estimator fit time per CV fold, not end-to-end workflow time.

| Model | Val. accuracy | Val. precision | Val. recall | Val. F1 | Val. ROC-AUC | Val. PR-AUC | CV accuracy | CV precision | CV recall | CV F1 | CV ROC-AUC | CV PR-AUC | Mean fold fit (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.8529 | 0.7020 | 0.3656 | 0.4808 | 0.8429 | 0.5946 | 0.8499 ± 0.0046 | 0.7035 ± 0.0226 | 0.3359 ± 0.0178 | 0.4547 ± 0.0205 | 0.8374 ± 0.0072 | 0.5958 ± 0.0182 | 3.426 |
| Logistic Regression | 0.8425 | 0.6327 | 0.3681 | 0.4654 | 0.8376 | 0.5851 | 0.8488 ± 0.0034 | 0.6685 ± 0.0167 | 0.3743 ± 0.0138 | 0.4798 ± 0.0141 | 0.8330 ± 0.0099 | 0.5911 ± 0.0163 | 0.573 |
| Support Vector Machine | 0.8438 | 0.6542 | 0.3418 | 0.4490 | 0.8354 | 0.5791 | 0.8477 ± 0.0036 | 0.6955 ± 0.0201 | 0.3247 ± 0.0160 | 0.4426 ± 0.0173 | 0.8313 ± 0.0095 | 0.5874 ± 0.0151 | 1.338 |
| Random Forest | 0.8240 | 0.8652 | 0.0653 | 0.1215 | 0.8239 | 0.5579 | 0.8205 ± 0.0014 | 0.9051 ± 0.0506 | 0.0407 ± 0.0065 | 0.0778 ± 0.0120 | 0.8261 ± 0.0067 | 0.5677 ± 0.0170 | 2.267 |
| Extra Trees | 0.8141 | 1.0000 | 0.0017 | 0.0034 | 0.8157 | 0.5473 | 0.8137 ± 0.0001 | 0.2000 ± 0.4000 | 0.0002 ± 0.0004 | 0.0004 ± 0.0008 | 0.8149 ± 0.0105 | 0.5511 ± 0.0188 | 2.106 |
| AdaBoost | 0.8269 | 0.6781 | 0.1340 | 0.2238 | 0.8062 | 0.5013 | 0.8316 ± 0.0016 | 0.7617 ± 0.0298 | 0.1412 ± 0.0131 | 0.2378 ± 0.0180 | 0.7986 ± 0.0094 | 0.5095 ± 0.0197 | 2.662 |
| Decision Tree | 0.8246 | 0.5452 | 0.3528 | 0.4284 | 0.7864 | 0.4729 | 0.8320 ± 0.0033 | 0.5907 ± 0.0137 | 0.3186 ± 0.0210 | 0.4137 ± 0.0203 | 0.7826 ± 0.0024 | 0.4863 ± 0.0167 | 0.785 |
| K-Nearest Neighbors | 0.8278 | 0.6802 | 0.1425 | 0.2356 | 0.7602 | 0.4561 | 0.8301 ± 0.0013 | 0.7207 ± 0.0228 | 0.1446 ± 0.0026 | 0.2408 ± 0.0043 | 0.7646 ± 0.0116 | 0.4844 ± 0.0186 | 0.176 |
| Gaussian Naive Bayes | 0.5444 | 0.2251 | 0.5920 | 0.3262 | 0.5784 | 0.2252 | 0.5245 ± 0.0069 | 0.2143 ± 0.0047 | 0.5824 ± 0.0230 | 0.3133 ± 0.0080 | 0.5677 ± 0.0074 | 0.2189 ± 0.0042 | 0.332 |
| DummyClassifier | 0.8137 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1863 | 0.8137 ± 0.0001 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.1863 ± 0.0001 | 0.180 |

All ten models completed. The high accuracy of the dummy model is a reminder that accuracy alone is misleading for this imbalanced target. HistGradientBoosting led the model zoo on CV ROC-AUC and PR-AUC; recall at threshold 0.50 was low, motivating the separate cost-based threshold analysis.

Full results: [model_comparison.csv](./model_comparison.csv), [phase2e_model_zoo_report.md](./phase2e_model_zoo_report.md), [model_comparison.png](./model_comparison.png).

## 5. Class imbalance comparison

Class weighting and fold-local SMOTENC were compared with no correction for Logistic Regression and Random Forest. Sampling was confined to each training fold; validation data were not resampled. The following validation values use threshold 0.50.

| Strategy | Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV ROC-AUC mean ± SD | FN | FP | Business cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No correction | Logistic Regression | 0.6327 | 0.3681 | 0.4654 | 0.8376 | 0.5851 | 0.8330 ± 0.0099 | 745 | 252 | ₹78,280,000 |
| Class weight | Logistic Regression | 0.4140 | 0.7354 | 0.5298 | 0.8363 | 0.5785 | 0.8310 ± 0.0094 | 312 | 1,227 | ₹49,605,000 |
| SMOTENC | Logistic Regression | 0.4255 | 0.6811 | 0.5238 | 0.8182 | 0.5508 | 0.8094 ± 0.0102 | 376 | 1,084 | ₹53,860,000 |
| No correction | Random Forest | 0.8652 | 0.0653 | 0.1215 | 0.8239 | 0.5579 | 0.8261 ± 0.0067 | 1,102 | 12 | ₹110,380,000 |
| Class weight | Random Forest | 0.4132 | 0.7150 | 0.5238 | 0.8235 | 0.5377 | 0.8238 ± 0.0089 | 336 | 1,197 | ₹51,555,000 |
| SMOTENC | Random Forest | 0.4330 | 0.5428 | 0.4817 | 0.7979 | 0.4732 | 0.7915 ± 0.0101 | 539 | 838 | ₹66,470,000 |

Class weighting substantially raised default recall and reduced validation cost in both model families, at the expense of more false positives and lower precision. For these experiments, Logistic Regression with class weights had the lowest cost at 0.50. This does not imply it is the final selected model: the final model/threshold comparison uses the tuned HistGradientBoosting candidate and the stated cost function.

Details: [imbalance_comparison.csv](./imbalance_comparison.csv), [phase2f_imbalance_report.md](./phase2f_imbalance_report.md), [imbalance_comparison.png](./imbalance_comparison.png).

## 6. Hyperparameter tuning

The two CV ROC-AUC leaders, HistGradientBoosting and Logistic Regression, were tuned using five-fold `GridSearchCV`, ROC-AUC scoring, and small parameter grids. Tuning remained on training data.

| Model | Tuned parameter(s) | Default CV ROC-AUC | Tuned CV ROC-AUC mean ± SD | Gain | Candidates | Tuning/refit time |
|---|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | `learning_rate=0.1`, `max_iter=100` | 0.8374 | 0.8394 ± 0.0061 | +0.0020 | 4 | 74.63 s |
| Logistic Regression | `C=0.5` | 0.8330 | 0.8339 ± 0.0101 | +0.0009 | 3 | 7.75 s |

The recorded baseline training times were 4.12 s for HistGradientBoosting and 0.58 s for Logistic Regression. The CV gain is modest; tuning supports candidate comparison rather than proving a large performance improvement. Candidate artifacts are stored under `models/candidates/`.

Details: [tuning_results.csv](./tuning_results.csv), [phase2g_tuning_report.md](./phase2g_tuning_report.md).

## 7. Final selected model and justification

**Selected candidate: tuned HistGradientBoosting.** It led the ten-model zoo in five-fold ROC-AUC and PR-AUC, retained the stronger tuned CV ROC-AUC of the two tuned candidates (0.8394 ± 0.0061 versus 0.8339 ± 0.0101), and achieved the lowest validation business cost among the evaluated final-model thresholds. Logistic Regression was close in ranking but had a higher cost at its own best threshold. The selection is a validation result and is not production certification.

The final pipeline was refitted on all 31,650 cleaned labeled training rows for the generated hold-out and scenario predictions. The fitted pipeline, threshold, and metadata are saved in `models/final_model.joblib`. The two tuned candidate pipelines are `models/candidates/histgradientboosting_tuned.joblib` and `models/candidates/logistic_regression_tuned.joblib`.

## 8. Optimal threshold and business cost

The threshold sweep in `threshold_analysis.csv` covers 0.05–0.95 in increments of 0.01. At the selected **0.11** threshold, a probability at or above 0.11 is flagged as default/review; below 0.11 is flagged non-default/review. On the 6,330-row validation set:

| Operating point | Threshold | False negatives | False positives | Validation cost | Cost/applicant | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional reference | 0.50 | 727 | 203 | ₹75,745,000 | ₹11,966.03 | 0.3834 | 0.6901 |
| Minimum observed cost | **0.11** | **138** | **2,045** | **₹44,475,000** | **₹7,026.07** | **0.8830** | **0.3373** |

At 0.11, 3,086 validation cases are predicted default and 3,244 non-default. Cost falls by ₹31,270,000 (41.28%) versus 0.50 because the false-negative cost is much larger. This is a material shift in the false-positive/false-negative trade-off; the threshold must not be interpreted as a policy approval without independent validation and business sign-off.

## 9. ROC, Precision–Recall, and calibration analysis

For the selected validation candidate, the recorded ROC-AUC is **0.8452**, PR-AUC / average precision is **0.6035**, and Brier score is **0.1080**. The ROC and precision–recall curves describe ranking performance; the calibration plot is a reliability diagnostic, and the Brier score alone does not establish adequate calibration. Validation labels were used to evaluate these quantities and the threshold, not to fit preprocessing or the estimator.

Charts: [roc_curves.png](./roc_curves.png), [precision_recall_curves.png](./precision_recall_curves.png), [calibration_curve.png](./calibration_curve.png), [business_cost_threshold.png](./business_cost_threshold.png), [selected_confusion_matrix.png](./selected_confusion_matrix.png).

## 10. Global feature importance

Permutation importance was measured on the validation partition as the decrease in ROC-AUC after permuting a raw feature. Top entries:

| Feature | Mean importance | Standard deviation |
|---|---:|---:|
| `credit_score` | 0.1265 | 0.0042 |
| `debt_to_income` | 0.0472 | 0.0039 |
| `late_payments_last_2y` | 0.0277 | 0.0011 |
| `has_collateral` | 0.0136 | 0.0018 |
| `branch_code` | 0.0055 | 0.0001 |
| `employer_category` | 0.0054 | 0.0010 |
| `years_at_current_job` | 0.0048 | 0.0004 |
| `loan_purpose` | 0.0043 | 0.0005 |
| `employment_type` | 0.0032 | 0.0004 |
| `residence_type` | 0.0030 | 0.0005 |

These are model-dependence measures, not causal effects or evidence that a feature is permissible. Full results: [permutation_importance.csv](./permutation_importance.csv), [feature_importance.png](./feature_importance.png).

## 11. Individual explanations: TC01, TC02, and TC08

Explanations use the final full-training model for scenario probabilities and one-feature-at-a-time replacement by a training reference value. Local changes are not additive or causal. Global importance in the final column is from the validation model.

| Case | Probability / threshold | Flag | Material local score sensitivities versus training reference |
|---|---:|---|---|
| TC01 | 0.0087 / 0.11 | Non-default / approve-review | Higher observed credit score (835 vs 716) corresponds to a −0.0194 probability change. No material one-feature risk-increasing change was recorded. |
| TC02 | 0.9298 / 0.11 | Default / reject-review | Lower credit score (545 vs 716): +0.3258; late payments (7 vs 0): +0.1642; utilization (1.05 vs 0.38): +0.0147. |
| TC08 | 0.0408 / 0.11 | Non-default / approve-review | Loan amount (₹1,800,000 vs ₹350,000 reference): +0.0172; Self-Employed vs Salaried: +0.0146; higher score (790 vs 716): −0.0556. |

The flag text is a model output for review, not a final approve/reject action. Full case details and the scenario source characteristics are in [scenario_explanations.csv](./scenario_explanations.csv) and [scenario_explanations.md](./scenario_explanations.md).

## 12. Final `loan_test.csv` prediction summary

`outputs/predictions_loan_test.csv` contains **8,000 rows** with `applicant_id`, `default_probability`, and `default_pred`. Persisted validation recorded source ID/order preservation, valid probabilities, and no missing values. The output has **3,965** predicted defaults and **4,035** predicted non-defaults; mean predicted default probability is **0.1873**. The test data has no labels, so these counts are predictions and cannot be scored for accuracy, recall, or cost.

## 13. Scenario prediction summary

`outputs/scenario_predictions.csv` contains all **12** scenario cases with their probability, binary flag, and threshold. **2** cases are flagged default and **10** non-default. TC01 = 0.0087, TC02 = 0.9298, and TC08 = 0.0408. Unknown/messy values are processed by the shared pipeline; no scenario-specific model was trained.

## 14. Model saving and deployment information

- Final serialized pipeline: `models/final_model.joblib`.
- Tuned candidates: `models/candidates/histgradientboosting_tuned.joblib` and `models/candidates/logistic_regression_tuned.joblib`.
- Prediction entry point: `predict.py`; it applies the saved pipeline to a raw CSV.
- The final artifact records the selected decision threshold and model metadata. Use the same threshold with this fitted model; changing the model or input distribution requires renewed evaluation.
- Workflow, CSVs, reports, and visualizations are orchestrated by `notebooks/task2_modelling.ipynb` and `src/phase2e_2j.py`. The notebook's execution cell runs the full workflow and can take several minutes; reviewing existing outputs does not require rerunning it.

## 15. Testing and implementation summary

The existing project report records **28 pytest tests passed**; the latest reported post-explanation update run completed in **36.55 seconds**, and Python source compilation succeeded. The tests cover the pipeline, leakage/target exclusion, imbalance/tuning/threshold helpers, prediction shapes/ranges, artifact round-trip, and scenario compatibility. The full model workflow runtime recorded in generated outputs is **399.49 seconds**. These are existing run results; no experiments or tests were rerun to prepare this report.

## 16. Limitations and future improvements

1. A random split and random five-fold CV do not establish future-cohort performance. Confirm application chronology, label maturity, and censoring, then evaluate time-aware and repeated/nested splits.
2. Model and threshold selection use the same validation partition, which can make the reported optimum optimistic. Use an independent calibration/policy-selection set or nested evaluation before deployment.
3. Validate probability calibration and cost assumptions with business owners. The threshold is sensitive to the false-negative/false-positive cost ratio.
4. Confirm whether the 12-month outcome is anchored to application or disbursal, and define the label cutoff and handling of non-disbursed/immature applications.
5. Confirm point-in-time validity of `debt_to_income`, availability and fairness implications of `branch_code`, and the permitted use of sensitive/proxy fields. The exclusion of `interest_rate` reflects unresolved timing, not a permanent conclusion.
6. Scenario cases are small and manually authored; successful processing is compatibility evidence, not a representative performance sample.
7. Test predictions have no ground-truth labels in this project, so their predictive quality is unknown.
8. The project does not establish production readiness, fair lending compliance, monitoring, or a human-review policy.

## 17. Final conclusion

The required 2E–2J model comparisons, imbalance experiments, tuning, threshold analysis, interpretation, saved model artifact, and test/scenario predictions are present. Tuned HistGradientBoosting is the current validation candidate with threshold 0.11 and validation cost ₹44,475,000. Its strong validation default recall (0.8830) comes with lower precision (0.3373) and many false positives (2,045), so the business trade-off must be explicitly accepted and independently validated.

The 0.11 threshold and generated predictions are not a production lending policy. Resolve target timing, feature governance, fairness, calibration, and independent validation before operational decisions.

## 18. Submission note and superseded report

Use this report as the consolidated 2E–2J summary. `phase3_model_evaluation.md` is a historical earlier-phase report and contains a different five-model workflow and threshold (0.40); it is superseded by the completed 2E–2J results in this report and should not be presented as the current final result without being clearly labeled historical.
