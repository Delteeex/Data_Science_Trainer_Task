# FinTrust Loan Default Prediction — Phase 2 Cleaning Report

## Scope and reproducibility

This phase implements deterministic data cleaning, schema validation, and reusable
scikit-learn preprocessing. Original CSV files were read-only inputs; no model was
trained, no predictions were generated, and no Phase 3 work was started.

Run from the project directory:

```powershell
python src\train.py
python -m pytest -q tests\test_pipeline.py
```

The notebook is `notebooks/task_ml.ipynb`, the existing notebook in the project.
Its nine code cells were executed end-to-end after the final cleaning changes.
The source `DS_Trainer_Task2_Data` directory is resolved relative to the project
location by `src/cleaning.py`.

## Dataset dimensions

| Dataset | Original rows × columns | Cleaned rows × columns | Notes |
|---|---:|---:|---|
| `loan_train.csv` | 32,600 × 35 | 31,650 × 37 | Exact duplicates and rows without usable targets omitted. |
| `loan_test.csv` | 8,000 × 34 | 8,000 × 36 | Target remains absent. |
| `loan_scenario_cases.csv` | 12 × 35 | 12 × 37 | All 12 profiles retained; scenario narrative retained but not used as a predictor. |

The three derived output columns are `application_year`, `application_month`
and `credit_score_no_history`. The raw `application_date` column is replaced
with year/month. Train/test/scenario
dimensions are consistent with their roles; the training-only `defaulted` target
is retained in cleaned training data.

## Cleaning operations

- Read all three original CSVs without writing back to them. Validate required
  columns, target placement, empty data, unexpected columns, and cross-file
  schema agreement.
- Normalize recognized target spellings and values to numeric `0`/`1`.
  Unrecognized nonmissing target values raise an error. Missing labels are
  retained during validation and omitted from the final training rows because
  they cannot be used as supervised examples.
- Drop exact duplicate training rows only. The source training file contains
  600 exact duplicates. After deduplication, 350 additional rows have missing
  target values and are omitted. No test or scenario applicants are dropped.
- Normalize documented category aliases, casing, whitespace, boolean spellings,
  numeric text, currency/unit representations, terms in years/months, and
  supported date formats. Application dates are converted to year and month.
- Convert invalid/unparseable numeric values and infinities to missing values
  for later pipeline imputation. Keep rows and flag suspicious values instead
  of removing valid applicants on the basis of an outlier rule.
- Preserve `NTC` / `New to Credit` as a missing numeric bureau score while
  deriving `credit_score_no_history`; this distinguishes known no-history
  applicants from an unreported score.
- Keep missing feature values in cleaned CSVs. Imputation is learned only when
  the scikit-learn preprocessor is fitted on a training fold: numeric median
  imputation with missing indicators and scaling; categorical constant
  imputation and one-hot encoding with unknown categories ignored.
- `src/train.py` provides a reproducible stratified split (`random_state=42`).
  The notebook fits preprocessing on `X_train` and transforms validation, test,
  and scenario rows with that fitted transformer; it does not fit preprocessing
  on the full dataset.

### Parsing decisions that require confirmation

- Ratios with an explicit percent sign, or an unmarked numeric value greater
  than 1.5, are divided by 100. Smaller unmarked values are treated as
  fractions. This retains the scenario's unmarked `1.05` as 105% utilization,
  but it is a parsing convention for ambiguous values, not a confirmed source
  system contract.
- `age`, bureau scores outside the documented 300–900 range, negative
  nonnegative measures, invalid terms, and invalid/unparseable values are
  normalized to missing where the parser can identify them. The scenario case
  TC11 has a negative requested loan amount: it is retained as a row, becomes
  missing in the cleaned numeric value, and is flagged by
  `validate_raw_application_values`. A caller must review validation flags
  before scoring; preprocessing alone is not business approval of invalid data.
- IQR bounds are diagnostic screening only. Outliers are not clipped or deleted.

## Predictor exclusions and timing

The feature selector excludes the following fields:

| Field | Reason |
|---|---|
| `defaulted` | Outcome label; never an input predictor. |
| `applicant_id` | Pure identifier retained for linkage, not prediction. |
| `recovery_calls_made` | Post-application recovery activity and direct temporal leakage. |
| `record_created_by` | Operational record metadata with proxy/leakage risk. |
| `country` | Constant (`India`) in the supplied datasets. |
| `loan_amount_lakhs` | Redundant alternate-unit copy of principal; Phase 1 found inconsistent scenario representations. |
| `interest_rate` | Offered rate's application-time availability is unconfirmed. Excluded pending clarification. |
| `scenario` | Hand-written scenario narrative, not a point-in-time application feature. |

The high-cardinality `branch_code` remains in the feature set; its deployment
availability, proxy/fairness risk, and stability need review before model
selection. `debt_to_income` also needs point-in-time provenance confirmation.
These decisions follow the available Phase 1 evidence and do not infer missing
column definitions.

## Validation results

- `python -m pytest -q tests\test_pipeline.py`: **15 passed**.
- `python src\train.py`: completed and wrote all three cleaned intermediate
  datasets; the command confirms that no estimator was fitted.
- Notebook: all **9/9 code cells** completed. Its fold-fitted preprocessor
  transformed validation, test, and all 12 scenario cases into 431 aligned
  encoded features: `(6,330, 431)`, `(8,000, 431)`, and `(12, 431)`.
- Cleaned row counts and target integrity:

| Check | Result |
|---|---:|
| Training target 0 (non-default) | 25,753 (81.37%) |
| Training target 1 (default) | 5,897 (18.63%) |
| Duplicate rows after training cleanup | 0 |
| Missing target labels in cleaned training output | 0 |
| Scenario rows retained | 12 |

Feature missingness remains by design: the cleaned files have 15,447 missing
cells in training, 3,936 in test, and 20 in scenarios. These are handled by
the pipeline at fit/transform time rather than by statistics computed over
the complete datasets.

## Generated files

- `outputs/cleaned/loan_train_cleaned.csv`
- `outputs/cleaned/loan_test_cleaned.csv`
- `outputs/cleaned/loan_scenario_cases_cleaned.csv`
- `outputs/phase2_cleaning_report.md` (this report)

No final model artifact or prediction file was created.

## Documentation and open questions

Appendix B and Section 5.3 are available in `Candidate_Task_Brief.pdf` (as
identified in the Phase 1 report). Appendix B supplies the field dictionary,
including the bureau score range and recovery activity definition. Section 5.3
requires the final pipeline to process all 12 scenarios and describes the
deployment input/CLI requirement. No missing appendix or section was identified.

Before Phase 3, confirm:

1. The source-system point-in-time definition of interest rate and
   debt-to-income, and whether either can change after the application decision.
2. The outcome observation cutoff/maturity policy for the 12-month default
   label and treatment of censored applications.
3. Whether `branch_code` is permitted and stable for deployment; evaluate
   sensitive/proxy fields and fairness constraints.
4. The intended conventions for ambiguous unmarked ratio values and sentinel
   values, including negative monthly EMI and zero/negative income.
5. Duplicate provenance and whether the 600 duplicated training rows should
   be removed under the approved record-identity policy.

## Phase 3 boundary

This report completes Phase 2 cleaning and preprocessing only. Model fitting,
model selection, cost-sensitive threshold selection, calibration, final test
predictions, and scenario decisions are deferred until explicit approval.
The stated error costs imply a 13.04% decision threshold as a mathematical
starting point only; any later operating threshold requires validation,
calibration, and policy review.
