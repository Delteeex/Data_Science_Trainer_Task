# Phase 1 - FinTrust Loan Default Dataset Analysis

## Scope and reproducibility

- Read the three original CSVs in `DS_Trainer_Task2_Data/` with `pandas.read_csv(..., low_memory=False)`. All row counts, nulls, vocabularies, numeric summaries and duplicate checks below are computed directly from those files; the source CSVs were not edited.
- Runtime used: Python 3.11.1; pandas 3.1.0; NumPy 2.4.6. CSV output uses UTF-8 with BOM for spreadsheet compatibility.
- The companion `phase1_data_quality.csv` contains one row per source column with exact observed categorical vocabularies (where applicable), missing counts, native-numeric summaries, IQR screening counts, and issue notes. IQR fences are a screening heuristic only and do not establish that a value is invalid.
- No model was trained, no predictions were generated, and no Phase 2 processing was started.

## Dataset overview

|File|Rows|Columns|Exact duplicate rows|Rows with >=1 missing field|Target present|
|---|---|---|---|---|---|
|loan_train.csv|32,600|35|600|11,550|Yes|
|loan_test.csv|8,000|34|0|2,781|No|
|loan_scenario_cases.csv|12|35|0|12|No|

### Schema and split consistency

- Training and hold-out test have the same 34 non-target feature columns; `defaulted` appears only in training. Scenario cases have the same underlying features plus a `scenario` narrative and omit `defaulted`. This matches the role described in the brief.
- `applicant_id` values do not overlap between train/test, train/scenario, or test/scenario. Train has 32,000 distinct IDs for 32,600 rows; the 600 excess rows are exact duplicate rows. Test and scenario IDs are unique.
- Native pandas dtypes are not consistent for every feature across the three files because the scenario file is hand-authored and deliberately includes dirty representations. The same issue is present within train/test string-valued numeric fields.

### loan_train.csv: columns, types, missingness and observed values

|Column|Loaded dtype|Missing|Unique non-null|Observed-value/profile note|
|---|---|---|---|---|
|applicant_id|str|0 (0.00%)|32,000|Identifier values are high-cardinality; examples shown; full ID list omitted.|
|application_date|str|0 (0.00%)|3,009|Date values are high-cardinality; examples shown; full date list omitted.|
|age|str|492 (1.51%)|92|Distinct values: 39, 32, 49, 21, 28, 44, 37, 33, 29, 40; (92 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|gender|str|0 (0.00%)|8|Distinct values: Male, F, MALE, Female, male, FEMALE, female, M|
|marital_status|str|0 (0.00%)|14|Distinct values: Married, Unmarried, Single, Divorced, M, S, married, single, Widowed, MARRIED; (14 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|dependents|str|653 (2.00%)|7|Distinct values: 1, 0, 3+, 2, 4, 3, 5|
|education|str|662 (2.03%)|24|Distinct values: M.Tech, HSC, Masters, Intermediate, Post Graduate, Doctorate, B.Tech, PG, 10th, Graduate; (24 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|employment_type|str|0 (0.00%)|19|Distinct values: Salaried, Self-Employed, SALARIED, Service, SE, Business Owner, Business, freelance, Freelancer, Salaried Employee; (19 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|employer_category|str|3,748 (11.50%)|6|Distinct values: MNC, Not Applicable, Private Ltd, Government, -, Startup|
|years_at_current_job|str|810 (2.48%)|515|Distinct values: 1.1, 3.6, 3.5, 1.0, 4.6, 6.8, 3.7, 1.8, 4.7, 5.5; (515 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|annual_income|str|961 (2.95%)|11,573|Distinct values: 1059600, 45900/month, 9.06 LPA, 840000, 3.1 lakh, 9.31 LPA, 11.42 LPA, 13.33 LPA, 14.15 LPA, 4.73 LPA; (11573 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|city_tier|str|0 (0.00%)|14|Distinct values: T1, 1, 2, Tier 3, tier 3, T3, 3, Metro, Tier 2, tier2; (14 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|residence_type|str|0 (0.00%)|14|Distinct values: Rented, Parental, Owned, Own, Family Owned, Parents, Company Provided, Rent, owned, rented; (14 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|years_at_residence|float64|0 (0.00%)|246|Numeric: min 0.000; Q1 2.900; median 4.600; mean 5.522; Q3 7.400; max 29.000; IQR flags 1068.|
|loan_purpose|str|0 (0.00%)|38|Distinct values: Appliances, Business Expansion, Personal, Wedding, Consumer Durables, Education, Travel, Medical, business, vehicle; (38 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|loan_amount|str|0 (0.00%)|5,668|Distinct values: 2751000, ₹156,000, 914000, 2.91L, 168000, 263000, ₹1,371,000, 3.28L, 440000, 630000; (5668 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|loan_term_months|str|0 (0.00%)|15|Distinct values: 24, 3 yrs, 36, 36 months, 24 months, 48 months, 60 months, 60, 48, 1 yrs; (15 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|interest_rate|str|0 (0.00%)|2,830|Distinct values: 0.1341, 14.56%, 15.42, 9.50, 15.57, 13.85, 12.50, 11.16, 11.12%, 12.29%; (2830 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|credit_score|str|1,602 (4.91%)|427|Distinct values: 665.0, 639.0, 609.0, 782.0, 706.0, 728.0, 999, 739.0, 732.0, 741.0; (427 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|num_credit_lines|int64|0 (0.00%)|12|Numeric: min 1; Q1 2.000; median 3.000; mean 3.507; Q3 4.000; max 12; IQR flags 438.|
|existing_monthly_emi|float64|0 (0.00%)|914|Numeric: min -1.000; Q1 0.000; median 2,800.000; mean 8,958.676; Q3 12,600.000; max 245,500.000; IQR flags 2213.|
|late_payments_last_2y|float64|638 (1.96%)|11|Numeric: min 0.000; Q1 0.000; median 0.000; mean 0.810; Q3 1.000; max 10.000; IQR flags 3075.|
|credit_utilization|str|0 (0.00%)|2,744|Distinct values: 13.1%, 0.371, 28.1, 18.9%, 0.209, 0.341, 0.297, 0.143, 0.671, 0.410; (2744 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|hard_inquiries_6m|int64|0 (0.00%)|9|Numeric: min 0; Q1 0.000; median 1.000; mean 1.170; Q3 2.000; max 8; IQR flags 67.|
|bank_relationship_months|float64|0 (0.00%)|266|Numeric: min 0.000; Q1 29.000; median 48.000; mean 56.868; Q3 76.000; max 300.000; IQR flags 1023.|
|has_cosigner|str|658 (2.02%)|12|Distinct values: false, no, No, 0, N, true, TRUE, Yes, FALSE, 1; (12 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|has_collateral|str|635 (1.95%)|12|Distinct values: yes, 0, No, false, true, no, Yes, TRUE, N, FALSE; (12 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|application_channel|str|0 (0.00%)|15|Distinct values: Online, Web, Branch, Partner App, Website, DSA Agent, Partner, online, Walk-in, branch; (15 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|branch_code|str|0 (0.00%)|960|Distinct values: BR319, BR165, BR-151, BR306, BR-145, BR317, BR115, BR008, BR224, BR179; (960 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|debt_to_income|str|2,615 (8.02%)|2,698|Distinct values: 148.7%, 0.117, 0.747, 19.1%, 0.334, 0.482, 0.242, 0.122, 0.849, 10.1%; (2698 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|recovery_calls_made|int64|0 (0.00%)|18|Numeric: min 0; Q1 0.000; median 0.000; mean 1.374; Q3 0.000; max 17; IQR flags 7142.|
|defaulted|str|361 (1.11%)|10|Distinct values: Default, 0, No, Non-Default, FALSE, N, TRUE, 1, Yes, Y|
|country|str|0 (0.00%)|1|Distinct values: India|
|loan_amount_lakhs|float64|0 (0.00%)|2,298|Numeric: min 0.250; Q1 1.950; median 3.480; mean 4.970; Q3 6.200; max 40.000; IQR flags 2100.|
|record_created_by|str|0 (0.00%)|4|Distinct values: ops_user_14, ops_user_22, system, sys_batch|

### loan_test.csv: columns, types, missingness and observed values

|Column|Loaded dtype|Missing|Unique non-null|Observed-value/profile note|
|---|---|---|---|---|
|applicant_id|str|0 (0.00%)|8,000|Identifier values are high-cardinality; examples shown; full ID list omitted.|
|application_date|str|0 (0.00%)|2,533|Date values are high-cardinality; examples shown; full date list omitted.|
|age|str|119 (1.49%)|88|Distinct values: 27, 32, 33, 51, 46, 37, 21, 50, 49, 44; (88 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|gender|str|0 (0.00%)|8|Distinct values: Female, Male, male, MALE, M, female, FEMALE, F|
|marital_status|str|0 (0.00%)|14|Distinct values: Married, Widowed, Single, Separated, Divorced, S, M, married, single, MARRIED; (14 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|dependents|str|158 (1.98%)|7|Distinct values: 0, 1, 2, 5, 3+, 3, 4|
|education|str|150 (1.88%)|24|Distinct values: Post Graduate, B.Com, PG, 10th, 12th, Graduate, M.Tech, Doctorate, BSc, SSC; (24 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|employment_type|str|0 (0.00%)|19|Distinct values: SALARIED, Self-Employed, Salaried, Freelancer, Self Employed, Freelancer , freelance, Salaried Employee, business owner, salaried; (19 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|employer_category|str|892 (11.15%)|6|Distinct values: Government, Not Applicable, Private Ltd, MNC, Startup, -|
|years_at_current_job|str|202 (2.53%)|387|Distinct values: 3.2, 1.5, 3.6, 2.1, 6.6, 10+, 1.0, 1.3, 9.3, 8.1; (387 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|annual_income|str|252 (3.15%)|5,095|Distinct values: ₹7,66,000, 733200, 481000, ₹5,18,239, ₹3,08,400, 8.39 LPA, 0, 337,200, 870000, 17.36 LPA; (5095 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|city_tier|str|0 (0.00%)|14|Distinct values: tier2, 1, Tier 2, 3, T2, T1, Tier 1, 2, Metro, Tier 3; (14 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|residence_type|str|0 (0.00%)|14|Distinct values: Family Owned, Owned, Rented, Company Provided, Parental, rented, Own, Rent, Self-Owned, Parents; (14 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|years_at_residence|float64|0 (0.00%)|214|Numeric: min 0.000; Q1 2.900; median 4.600; mean 5.523; Q3 7.500; max 25.000; IQR flags 250.|
|loan_purpose|str|0 (0.00%)|38|Distinct values: Home Renovation, vehicle, Personal, Debt Consolidation, Education, Business, Wedding, Edu Loan, Other, Misc; (38 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|loan_amount|str|0 (0.00%)|3,197|Distinct values: 916000, 6.60L, 537000, 2.41L, 1.02L, 6.82L, 661000, 253000, 4.84L, 7.32L; (3197 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|loan_term_months|str|0 (0.00%)|15|Distinct values: 60, 12, 60 months, 24, 2 yrs, 48, 3 yrs, 36, 36 months, 4 yrs; (15 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|interest_rate|str|0 (0.00%)|2,197|Distinct values: 16.34, 14.46, 0.0950, 9.50, 10.17, 14.27, 11.79%, 12.18%, 14.31, 14.00; (2197 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|credit_score|str|443 (5.54%)|389|Distinct values: 511.0, 643.0, 778.0, 899.0, 813.0, 762.0, 681.0, 720.0, 680.0, 632.0; (389 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|num_credit_lines|int64|0 (0.00%)|11|Numeric: min 1; Q1 2.000; median 3.000; mean 3.495; Q3 4.000; max 12; IQR flags 107.|
|existing_monthly_emi|float64|0 (0.00%)|621|Numeric: min -1.000; Q1 0.000; median 2,700.000; mean 8,846.487; Q3 12,600.000; max 198,400.000; IQR flags 540.|
|late_payments_last_2y|float64|171 (2.14%)|10|Numeric: min 0.000; Q1 0.000; median 0.000; mean 0.822; Q3 1.000; max 11.000; IQR flags 779.|
|credit_utilization|str|0 (0.00%)|2,207|Distinct values: 0.453, 53.4%, 0.120, 0.029, 0.088, 0.771, 0.503, 0.388, 46.7%, 84.3%; (2207 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|hard_inquiries_6m|int64|0 (0.00%)|9|Numeric: min 0; Q1 0.000; median 1.000; mean 1.156; Q3 2.000; max 8; IQR flags 14.|
|bank_relationship_months|float64|0 (0.00%)|222|Numeric: min 1.000; Q1 29.000; median 50.000; mean 57.117; Q3 76.000; max 283.000; IQR flags 234.|
|has_cosigner|str|159 (1.99%)|12|Distinct values: N, No, FALSE, 0, no, false, Yes, yes, TRUE, Y; (12 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|has_collateral|str|177 (2.21%)|12|Distinct values: Yes, yes, true, no, FALSE, 0, false, No, 1, N; (12 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|application_channel|str|0 (0.00%)|15|Distinct values: Branch, Partner, Online, Partner App, Web, dsa agent, DSA Agent, Walk-in, online, DSA; (15 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|branch_code|str|0 (0.00%)|857|Distinct values: BR193, BR074, br245, BR140, BR025, br314, BR010, BR207, BR202, BR033; (857 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|debt_to_income|str|604 (7.55%)|1,938|Distinct values: 0.972, 0.281, 48.9%, 0.083, 0.451, 31.3%, 42.3%, 0.345, 0.38, 16.0%; (1938 distinct; full category vocabulary in CSV (ID/date/narrative lists omitted))|
|recovery_calls_made|int64|0 (0.00%)|18|Numeric: min 0; Q1 0.000; median 0.000; mean 1.387; Q3 0.000; max 20; IQR flags 1756.|
|country|str|0 (0.00%)|1|Distinct values: India|
|loan_amount_lakhs|float64|0 (0.00%)|1,557|Numeric: min 0.250; Q1 2.010; median 3.520; mean 5.043; Q3 6.280; max 40.000; IQR flags 536.|
|record_created_by|str|0 (0.00%)|4|Distinct values: ops_user_14, ops_user_22, system, sys_batch|

### loan_scenario_cases.csv: columns, types, missingness and observed values

|Column|Loaded dtype|Missing|Unique non-null|Observed-value/profile note|
|---|---|---|---|---|
|applicant_id|str|0 (0.00%)|12|Identifier values are high-cardinality; examples shown; full ID list omitted.|
|scenario|str|0 (0.00%)|12|Scenario narratives are descriptive, not features; full narrative list omitted.|
|application_date|str|0 (0.00%)|1|Date values are high-cardinality; examples shown; full date list omitted.|
|age|str|1 (8.33%)|3|Distinct values: 34, 23, 41 yrs|
|gender|str|0 (0.00%)|1|Distinct values: Female|
|marital_status|str|0 (0.00%)|2|Distinct values: Married, Single|
|dependents|int64|0 (0.00%)|2|Numeric: min 0; Q1 1.000; median 1.000; mean 0.917; Q3 1.000; max 1; IQR flags 1.|
|education|str|1 (8.33%)|2|Distinct values: Graduate, MBA|
|employment_type|str|0 (0.00%)|3|Distinct values: Salaried, Freelancer, Self-Employed|
|employer_category|str|0 (0.00%)|3|Distinct values: Government, Not Applicable, MNC|
|years_at_current_job|float64|1 (8.33%)|3|Numeric: min 0.500; Q1 6.000; median 6.000; mean 6.045; Q3 6.000; max 12.000; IQR flags 2.|
|annual_income|str|0 (0.00%)|5|Distinct values: 900000, 480000, 75000/month, ₹12,50,000, 700000|
|city_tier|str|0 (0.00%)|2|Distinct values: 1, Tier-2|
|residence_type|str|0 (0.00%)|1|Distinct values: Owned|
|years_at_residence|int64|0 (0.00%)|1|Numeric: min 5; Q1 5.000; median 5.000; mean 5.000; Q3 5.000; max 5; IQR flags 0.|
|loan_purpose|str|0 (0.00%)|3|Distinct values: Home Renovation, Debt Consolidation, Crypto Investment|
|loan_amount|str|0 (0.00%)|4|Distinct values: 400000, 6.5L, 1800000, -250000|
|loan_term_months|str|0 (0.00%)|3|Distinct values: 36, 5 yrs, 24|
|interest_rate|str|0 (0.00%)|3|Distinct values: 11.5, 21.0, 12.75%|
|credit_score|float64|1 (8.33%)|4|Numeric: min 545.000; Q1 790.000; median 790.000; mean 774.545; Q3 790.000; max 835.000; IQR flags 3.|
|num_credit_lines|int64|0 (0.00%)|1|Numeric: min 3; Q1 3.000; median 3.000; mean 3.000; Q3 3.000; max 3; IQR flags 0.|
|existing_monthly_emi|int64|0 (0.00%)|2|Numeric: min 0; Q1 0.000; median 0.000; mean 3,166.667; Q3 0.000; max 38,000; IQR flags 1.|
|late_payments_last_2y|float64|1 (8.33%)|2|Numeric: min 0.000; Q1 0.000; median 0.000; mean 0.636; Q3 0.000; max 7.000; IQR flags 1.|
|credit_utilization|str|1 (8.33%)|3|Distinct values: 0.22, 1.05, 35%|
|hard_inquiries_6m|int64|0 (0.00%)|2|Numeric: min 0; Q1 0.000; median 0.000; mean 0.417; Q3 0.000; max 5; IQR flags 1.|
|bank_relationship_months|int64|0 (0.00%)|2|Numeric: min 6; Q1 84.000; median 84.000; mean 77.500; Q3 84.000; max 84; IQR flags 1.|
|has_cosigner|str|1 (8.33%)|3|Distinct values: N, Yes, Y|
|has_collateral|str|0 (0.00%)|3|Distinct values: N, TRUE, Y|
|application_channel|str|0 (0.00%)|2|Distinct values: Online, DSA Agent|
|branch_code|str|0 (0.00%)|2|Distinct values: BR015, BR999|
|debt_to_income|float64|12 (100.00%)|0|All values missing|
|recovery_calls_made|int64|0 (0.00%)|2|Numeric: min 0; Q1 0.000; median 0.000; mean 0.750; Q3 0.000; max 9; IQR flags 1.|
|country|str|0 (0.00%)|1|Distinct values: India|
|loan_amount_lakhs|float64|0 (0.00%)|1|Numeric: min 4.000; Q1 4.000; median 4.000; mean 4.000; Q3 4.000; max 4.000; IQR flags 0.|
|record_created_by|str|0 (0.00%)|1|Distinct values: system|

## Target analysis

- The training target is stored as text with ten observed encodings plus nulls, rather than a canonical binary type. Under the explicit case-insensitive mapping `1/true/yes/y/Default -> default (1)` and `0/false/no/n/Non-Default -> non-default (0)`, all non-null tokens are recognized; confirm the mapping with the data owner before cleaning.
- Labeled records: **32,239** of 32,600 (361 missing target labels; 1.11% of rows). Positive/default: **6,007 (18.63% of labeled)**. Non-default: **26,232 (81.37% of labeled)**. Ratio non-default:default is **4.37:1**. Thus classes are moderately imbalanced; default is the minority class.
- Raw target counts: `0` = 15,691; `No` = 3,967; `1` = 3,566; `FALSE` = 2,658; `N` = 2,604; `Non-Default` = 1,312; `Yes` = 921; `Y` = 617; `TRUE` = 587; `nan` = 361; `Default` = 316.
- `defaulted` is absent from both `loan_test.csv` and `loan_scenario_cases.csv`; the latter should be prediction-only profiles, not labeled validation data.

## Numerical distributions and data-quality observations

- The detailed native-numeric distributions and IQR-flag counts are listed per column above. Most economically meaningful numeric fields (`age`, income, requested amount, term, rate, credit score, utilization and debt-to-income) are loaded as strings because formats/units vary; they are therefore intentionally not presented as a single numeric distribution before defining safe unit normalization.


### Diagnostic distributions for numeric-looking text columns

The next table applies display-only unit parsing to raw string fields (income: monthly x12, lakh/LPA x100,000, K x1,000; principal: INR/K/L to INR; term: years to months; rate/ratios: fraction-to-percent where value <= 1 and no percent sign). It is not a cleaning decision. `parseable/nonmissing` shows coverage; values remain unsaved and no source file was altered. Ambiguous tokens such as `<1`, `10+`, and `NTC` are not coerced into numeric values.

|Dataset|Field|Parsed/nonmissing|Min; Q1; median; mean; Q3; max|
|---|---|---:|---|
|loan_train.csv|`age`|32,108/32,108|-1.000; 30.000; 36.000; 36.904; 42.000; 999.000|
|loan_train.csv|`years_at_current_job`|31,112/31,790|0.000; 2.200; 4.100; 6.281; 6.800; 259.000|
|loan_train.csv|`annual_income`|31,639/31,639|-1.000; 407,270.000; 636,000.000; 836,605.740; 986,000.000; 99,999,999.000|
|loan_train.csv|`loan_amount`|32,600/32,600|25,000.000; 195,000.000; 348,000.000; 496,978.712; 620,000.000; 4,000,000.000|
|loan_train.csv|`loan_term_months`|32,600/32,600|12.000; 24.000; 36.000; 37.295; 48.000; 60.000|
|loan_train.csv|`interest_rate`|32,600/32,600|9.500; 10.580; 12.350; 12.550; 14.140; 23.210|
|loan_train.csv|`credit_score`|30,435/30,998|-1.000; 667.000; 715.000; 709.334; 763.000; 999.000|
|loan_train.csv|`credit_utilization`|32,600/32,600|1.000; 23.200; 37.900; 39.325; 53.900; 111.700|
|loan_train.csv|`debt_to_income`|29,985/29,985|0.700; 20.000; 35.100; 39.066; 53.900; 605.700|
|loan_test.csv|`age`|7,881/7,881|-1.000; 30.000; 36.000; 36.543; 43.000; 999.000|
|loan_test.csv|`years_at_current_job`|7,656/7,798|0.000; 2.300; 4.100; 6.137; 7.000; 240.000|
|loan_test.csv|`annual_income`|7,748/7,748|-1.000; 415,200.000; 650,000.000; 865,444.471; 989,250.000; 99,999,999.000|
|loan_test.csv|`loan_amount`|8,000/8,000|25,000.000; 201,000.000; 352,000.000; 504,289.750; 628,000.000; 4,000,000.000|
|loan_test.csv|`loan_term_months`|8,000/8,000|12.000; 24.000; 36.000; 37.219; 48.000; 60.000|
|loan_test.csv|`interest_rate`|8,000/8,000|9.500; 10.580; 12.380; 12.557; 14.100; 24.940|
|loan_test.csv|`credit_score`|7,432/7,557|-1.000; 666.000; 715.000; 708.806; 763.000; 999.000|
|loan_test.csv|`credit_utilization`|8,000/8,000|1.000; 23.600; 38.000; 39.360; 53.900; 108.100|
|loan_test.csv|`debt_to_income`|7,396/7,396|1.001; 20.300; 34.900; 39.345; 54.200; 268.900|
|loan_scenario_cases.csv|`age`|11/11|23.000; 34.000; 34.000; 33.636; 34.000; 41.000|
|loan_scenario_cases.csv|`years_at_current_job`|11/11|0.500; 6.000; 6.000; 6.045; 6.000; 12.000|
|loan_scenario_cases.csv|`annual_income`|12/12|480,000.000; 850,000.000; 900,000.000; 860,833.333; 900,000.000; 1,250,000.000|
|loan_scenario_cases.csv|`loan_amount`|12/12|-250,000.000; 400,000.000; 400,000.000; 600,000.000; 462,500.000; 1,800,000.000|
|loan_scenario_cases.csv|`loan_term_months`|12/12|24.000; 36.000; 36.000; 36.000; 36.000; 60.000|
|loan_scenario_cases.csv|`interest_rate`|12/12|11.500; 11.500; 11.500; 12.396; 11.500; 21.000|
|loan_scenario_cases.csv|`credit_score`|11/11|545.000; 790.000; 790.000; 774.545; 790.000; 835.000|
|loan_scenario_cases.csv|`credit_utilization`|11/11|1.050; 22.000; 22.000; 21.277; 22.000; 35.000|
|loan_scenario_cases.csv|`debt_to_income`|0/0|no numeric values|

- Recurrent categorical variants include casing, abbreviations, synonyms and spelling differences in `gender`, `marital_status`, `education`, `employment_type`, `city_tier`, `residence_type`, `loan_purpose`, and `application_channel`. Boolean flags use `Y/Yes/true/1` and `N/No/false/0` variants. Full vocabularies for these columns are in the quality CSV.
- Missingness is concentrated in `employer_category` (~11%), `debt_to_income` (~8%), `credit_score` (~5%), and smaller gaps in income, age, dependents, education, employment tenure, late payments, and cosigner/collateral flags. Scenario `debt_to_income` is entirely missing (12/12).
- Numeric IQR flags are not automatically errors: for example, the zero-inflated `recovery_calls_made` has an upper fence of 0, and discrete credit/late-payment fields naturally produce IQR flags. Review with business semantics rather than mechanically clipping/removing.

### Notable defects and checks

|File|Exact duplicate rows|Repeated applicant IDs|Rows with any missing|Total rows|
|---|---|---|---|---|
|loan_train.csv|600|600|11,550|32,600|
|loan_test.csv|0|0|2,781|8,000|
|loan_scenario_cases.csv|0|0|12|12|

- `loan_train.csv` contains 600 exact duplicate rows and 600 repeated applicant IDs; review duplicate provenance before any split or training so copies cannot straddle validation folds.
- `existing_monthly_emi` contains negative sentinel values: 54 in train and 8 in test (the observed minimum is -1). Do not treat -1 as a real EMI without a documented sentinel rule.
- Appendix B defines bureau score range 300-900, while raw numeric score values include out-of-range sentinel-like values such as `999`; `NTC` / `New to Credit` also represent no history and should not be converted to a fabricated score.
- Unit-aware screening of mixed numeric text found annual income values of -1 (29 train, 2 test), zeros (20 train, 6 test), and an extreme raw value of 99,999,999; confirm sentinel/zero meanings rather than treating these as ordinary income. Age screening flags 57 train and 20 test values outside 0-120, including negative and sentinel-like entries; these are screening bounds, not documented business rules.
- After diagnostic-only percentage normalization (explicit percent strings as written; fractional values scaled by 100), credit utilization exceeds 100% in 10 train and 5 test rows. Debt-to-income exceeds 100% in 372 train and 88 test rows, with maxima 605.7% and 268.9%; these may reflect genuinely high obligations or formatting/source errors and require business review.
- Scenario rows deliberately demonstrate invalid/heterogeneous inputs, including a negative loan amount (TC11), values with embedded currency/units, an age with a suffix, missing fields, inconsistent city-tier formats, and unseen categories/branch codes. The scenarios should be used later as input robustness cases, not silently coerced.
- Application dates parse in all three files. Train/test cover 2022-01-01 to 2024-12-09; scenario dates are all 2024-06-15. Multiple date string formats occur, so consistent date parsing is required.
- `loan_amount` and `loan_amount_lakhs` appear intended as duplicate representations. After handling INR, K, L/lakh and currency formatting, all train/test rows agree within Rs 1; four scenario rows do not (TC07-TC09 and TC11).
- `country` is constant (`India`) in each dataset. `branch_code` is high-cardinality (960 training values, 857 test values) and should be checked for support/overfit or operational proxies. `record_created_by` is operational metadata, not applicant information.

## Leakage, timing, identifiers and split risks

| Field | Phase 1 assessment | Evidence / required decision |
|---|---|---|
| `recovery_calls_made` | **Strong direct target leakage; exclude from application-time model features.** | Recovery-team calls occur after origination/default risk materializes. In labeled training rows, median is 7 for defaults vs 0 for non-defaults (means 7.02 vs 0.08). The association is consistent with a post-outcome field and it is not available at application.
| `applicant_id` | Identifier, not a feature. | Unique/repeated ID behavior is recorded above; use it only for row linkage/output, not prediction.
| `application_date` | Available at decision time, but validate split strategy and outcome maturity. | The label is measured over a 12-month horizon, so late applications need complete outcome observation; obtain the cohort/label-extraction cutoff and use time-aware validation if appropriate.
| `interest_rate` | Timing ambiguous. | Appendix B says annual rate "offered"; confirm if determined before the decision or only after approval/underwriting.
| `debt_to_income` | Potentially valid but needs point-in-time provenance. | Appendix B says it is captured by the origination system; confirm it is computed from applicant-time information and not updated after decision.
| `loan_amount_lakhs` | Likely engineered duplicate of `loan_amount`. | Both describe principal in different units; standardize and verify equality, then avoid duplicate/inconsistent versions.
| `record_created_by`, `branch_code` | Operational/source metadata or high-cardinality proxy risk. | Confirm deployment availability and permitted use; evaluate leakage, stability, and fairness before deciding whether to retain.
| `country` | Constant in supplied data. | All values are India; it adds no variation in these files.

### Train-test category support

- The structural feature schema is consistent, but raw category vocabularies have train-only and test-only values. Examples include `branch_code` values in training not represented in test, and variable-format values in numeric-like columns. The final pipeline must handle unseen values without errors; do not infer missing categories from the holdout labels.
- The three files have no applicant ID overlap, but use of an ordinary random split could still be misleading if applications are time-ordered or repeated applicants occur in raw historical records. Resolve split design after understanding label timing.

## Business-cost implications (threshold planning only)

- Given the stated costs, approving an applicant with default probability `p` has expected cost `INR 100000 * p`; rejecting has expected cost `INR 15000 * (1 - p)`. With calibrated probabilities and these costs alone, reject when `p > 15,000 / (100,000 + 15,000) = **0.1304** (13.04%). This is a mathematical starting point, not a fitted or approved operating threshold.
- Threshold selection must use out-of-fold/validation probabilities, calibration checks and the documented cost assumptions; account for policy constraints and any costs omitted from the brief. Do not select a threshold on the hidden test labels.

## Documentation status

- **Appendix B is available** in `Candidate_Task_Brief.pdf` (page 14, "Appendix B - Task 2 data dictionary"). It documents field meanings, including `recovery_calls_made`, score range, debt-to-income and target definition.
- **Section 5.3 is available** in the same PDF (page 10, "5.3 Scenario cases & deployment"). It requires processing all 12 scenario rows through the final saved pipeline, writing scenario predictions, and providing a CLI that accepts raw/dirty CSV input.
- The brief defines the event as 90+ days past due within 12 months **of disbursal**, while the task request phrases the horizon as within 12 months **of application**. Clarify which timestamp anchors the target and how applications that are not disbursed / have insufficient follow-up are labeled.
- Additional business/process facts needed before feature decisions: exact as-of timestamp for each field (especially recovery calls, offered interest rate, DTI); target-label extraction date and maturity/censoring policy; whether duplicate applications are legitimate records; canonical normalization for all units/category synonyms; and permitted/prohibited feature use for IDs, branch, gender and other sensitive/proxy fields.

## Recommended Phase 2 next steps (do not execute until approved)

1. Confirm the application-versus-disbursal label horizon, label observation cutoff, and censored/unlabeled record handling.
2. Establish point-in-time feature eligibility; remove `recovery_calls_made` and `applicant_id` from predictors, and resolve `interest_rate`, `debt_to_income`, and operational metadata.
3. Define auditable parsers/unit rules for income, principal (INR/K/L), rate, ratios, term, age/tenure, bureau `NTC`, booleans and categories; validate against the 12 scenario cases and raise/flag impossible inputs such as negative principal.
4. Investigate duplicate applications and apply deduplication only under a documented key/policy; design time-aware, applicant-safe validation after checking outcome maturity.
5. Build preprocessing/model pipelines that tolerate missing and unseen categories; establish cost-sensitive threshold and probability calibration on validation data. Keep test labels untouched and do not train on scenario descriptions.
6. Add tests for output shape/type, deterministic inference, invalid-input handling, serialization round trips and all 12 scenarios, then compare validation metrics and expected business cost.

**Phase 1 ends here. No Phase 2 actions or model training were performed.**
