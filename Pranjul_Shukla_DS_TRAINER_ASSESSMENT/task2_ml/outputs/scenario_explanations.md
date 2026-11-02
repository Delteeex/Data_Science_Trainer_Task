# Individual scenario explanations

The exact source identifier is `applicant_id`; TC01, TC02, and TC08 were selected from `loan_scenario_cases.csv` (TC08 is not TC09). Probabilities and local perturbations use the selected tuned pipeline refitted on all labeled training records. Global permutation importance uses the separate 80%-fit validation model and untouched validation data.

Local explanations perturb one cleaned predictor at a time to its training-partition median (numeric) or mode (categorical) and report the change in predicted probability relative to the observed value. The case tables highlight one-at-a-time score changes of at least 0.005; smaller changes are omitted from the narrative to avoid over-interpreting negligible effects. They are descriptive model sensitivities, not counterfactual causal effects. Field meanings and profile statements come only from the source records. Globally important fields are those whose permutation decreased validation ROC-AUC; this is not evidence of causation.

## Global validation permutation importance

| Feature                  | Permutation Importance Mean | Permutation Importance Std |
| ------------------------ | --------------------------- | -------------------------- |
| credit_score             | 0.1265                      | 0.0042                     |
| debt_to_income           | 0.0472                      | 0.0039                     |
| late_payments_last_2y    | 0.0277                      | 0.0011                     |
| has_collateral           | 0.0136                      | 0.0018                     |
| branch_code              | 0.0055                      | 0.0001                     |
| employer_category        | 0.0054                      | 0.0010                     |
| years_at_current_job     | 0.0048                      | 0.0004                     |
| loan_purpose             | 0.0043                      | 0.0005                     |
| employment_type          | 0.0032                      | 0.0004                     |
| residence_type           | 0.0030                      | 0.0005                     |
| hard_inquiries_6m        | 0.0028                      | 0.0011                     |
| loan_term_months         | 0.0028                      | 0.0007                     |
| bank_relationship_months | 0.0026                      | 0.0011                     |
| application_channel      | 0.0024                      | 0.0009                     |
| has_cosigner             | 0.0022                      | 0.0007                     |

## Cases

### TC01

- Applicant ID: `TC01`
- Scenario description (from source): Prime borrower: high score, low obligations, government employee
- Default probability from the selected final full-training model: 0.0087
- Selected threshold: 0.11
- Binary prediction at the validation-selected threshold: 0
- Decision interpretation: flagged non-default / approve-review
- Source characteristics: {'age': '34', 'employment_type': 'Salaried', 'annual_income': '900000', 'loan_amount': '400000', 'loan_term_months': '36', 'credit_score': '835.0', 'late_payments_last_2y': '0.0', 'credit_utilization': '0.22', 'existing_monthly_emi': '0', 'has_collateral': 'N', 'has_cosigner': 'N', 'debt_to_income': None, 'years_at_current_job': '12.0'}
- Locally risk-increasing relative to one-feature-at-a-time training median/mode replacement: No single-feature perturbation moved the score in this direction.
- Locally risk-decreasing relative to that reference: `credit_score` (observed np.float64(835.0); score delta -0.0194; global permutation importance 0.12655)

These are model-score perturbations, not causal explanations; changes are one feature at a time and are not additive. Global importance is validation ROC-AUC decrease after permuting the raw feature.

| Feature      | Observed cleaned value | Training reference value | Probability delta vs reference | Global permutation importance |
| ------------ | ---------------------- | ------------------------ | ------------------------------ | ----------------------------- |
| credit_score | 835.0000               | 716.0000                 | -0.0194                        | 0.1265                        |

### TC02

- Applicant ID: `TC02`
- Scenario description (from source): Very high risk: low score, many late payments, maxed-out utilisation, DSA channel
- Default probability from the selected final full-training model: 0.9298
- Selected threshold: 0.11
- Binary prediction at the validation-selected threshold: 1
- Decision interpretation: flagged default / reject-review
- Source characteristics: {'age': '34', 'employment_type': 'Freelancer', 'annual_income': '480000', 'loan_amount': '400000', 'loan_term_months': '36', 'credit_score': '545.0', 'late_payments_last_2y': '7.0', 'credit_utilization': '1.05', 'existing_monthly_emi': '38000', 'has_collateral': 'N', 'has_cosigner': 'N', 'debt_to_income': None, 'years_at_current_job': '6.0'}
- Locally risk-increasing relative to one-feature-at-a-time training median/mode replacement: `credit_score` (observed np.float64(545.0); score delta +0.3258; global permutation importance 0.12655); `late_payments_last_2y` (observed np.float64(7.0); score delta +0.1642; global permutation importance 0.02767); `credit_utilization` (observed np.float64(1.05); score delta +0.0147; global permutation importance 0.00206)
- Locally risk-decreasing relative to that reference: `residence_type` (observed 'Owned'; score delta -0.0178; global permutation importance 0.00299); `years_at_current_job` (observed np.float64(6.0); score delta -0.0118; global permutation importance 0.00483); `bank_relationship_months` (observed np.float64(84.0); score delta -0.0069; global permutation importance 0.00264)

These are model-score perturbations, not causal explanations; changes are one feature at a time and are not additive. Global importance is validation ROC-AUC decrease after permuting the raw feature.

| Feature               | Observed cleaned value | Training reference value | Probability delta vs reference | Global permutation importance |
| --------------------- | ---------------------- | ------------------------ | ------------------------------ | ----------------------------- |
| credit_score          | 545.0                  | 716.0                    | 0.3258                         | 0.1265                        |
| late_payments_last_2y | 7.0                    | 0.0                      | 0.1642                         | 0.0277                        |
| residence_type        | Owned                  | Rented                   | -0.0178                        | 0.0030                        |
| credit_utilization    | 1.05                   | 0.38                     | 0.0147                         | 0.0021                        |
| existing_monthly_emi  | 38000.0                | 2800.0                   | 0.0140                         | 0.0004                        |
| years_at_current_job  | 6.0                    | 4.0                      | -0.0118                        | 0.0048                        |
| hard_inquiries_6m     | 5.0                    | 1.0                      | 0.0092                         | 0.0028                        |
| application_channel   | DSA Agent              | Branch                   | 0.0073                         | 0.0024                        |

### TC08

- Applicant ID: `TC08`
- Scenario description (from source): Self-employed WITHOUT collateral, high loan-to-income
- Default probability from the selected final full-training model: 0.0408
- Selected threshold: 0.11
- Binary prediction at the validation-selected threshold: 0
- Decision interpretation: flagged non-default / approve-review
- Source characteristics: {'age': '34', 'employment_type': 'Self-Employed', 'annual_income': '700000', 'loan_amount': '1800000', 'loan_term_months': '24', 'credit_score': '790.0', 'late_payments_last_2y': '0.0', 'credit_utilization': '0.22', 'existing_monthly_emi': '0', 'has_collateral': 'N', 'has_cosigner': 'N', 'debt_to_income': None, 'years_at_current_job': '6.0'}
- Locally risk-increasing relative to one-feature-at-a-time training median/mode replacement: `loan_amount` (observed np.float64(1800000.0); score delta +0.0172; global permutation importance 0.00146); `employment_type` (observed 'Self-Employed'; score delta +0.0146; global permutation importance 0.00321)
- Locally risk-decreasing relative to that reference: `credit_score` (observed np.float64(790.0); score delta -0.0556; global permutation importance 0.12655); `loan_purpose` (observed 'Home Renovation'; score delta -0.0213; global permutation importance 0.00431); `bank_relationship_months` (observed np.float64(84.0); score delta -0.0098; global permutation importance 0.00264)

These are model-score perturbations, not causal explanations; changes are one feature at a time and are not additive. Global importance is validation ROC-AUC decrease after permuting the raw feature.

| Feature                  | Observed cleaned value | Training reference value | Probability delta vs reference | Global permutation importance |
| ------------------------ | ---------------------- | ------------------------ | ------------------------------ | ----------------------------- |
| credit_score             | 790.0                  | 716.0                    | -0.0556                        | 0.1265                        |
| loan_purpose             | Home Renovation        | Debt Consolidation       | -0.0213                        | 0.0043                        |
| loan_amount              | 1800000.0              | 350000.0                 | 0.0172                         | 0.0015                        |
| employment_type          | Self-Employed          | Salaried                 | 0.0146                         | 0.0032                        |
| bank_relationship_months | 84.0                   | 48.0                     | -0.0098                        | 0.0026                        |
| hard_inquiries_6m        | 0.0                    | 1.0                      | -0.0067                        | 0.0028                        |
| residence_type           | Owned                  | Rented                   | -0.0063                        | 0.0030                        |

