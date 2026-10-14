# Credit Risk Modeling – Probability of Default

Predicts whether a borrower defaults (`loan_status = 1`) using the Kaggle
*Credit Risk Dataset* (32k loans). Builds and compares three models:

| Model | Role |
|---|---|
| Logistic regression on WOE features | interpretable baseline |
| LightGBM (random-search tuned, early stopping) | best accuracy, explained with SHAP |
| Scorecard (600 pts @ 50:1 odds, PDO 20) | regulatory-style points table |

## Project layout
```
credit_risk_project/
├── data/credit_risk_dataset.csv
├── src/
│   ├── config.py          # paths, constants, cost matrix
│   ├── data_cleaning.py   # stage A (rules) + stage B (train-fitted caps/medians)
│   ├── features.py        # feature engineering
│   ├── woe.py             # WOE / IV encoder
│   ├── models.py          # logreg, LightGBM tuning, scorecard
│   ├── evaluate.py        # AUC/Gini/KS, cutoff, plots, SHAP
│   └── predict.py         # score one applicant + reason codes
├── main.py                # run the whole pipeline
├── app.py                 # Streamlit demo
├── models/                # saved bundle (created by main.py)
└── reports/               # tables and plots (created by main.py)
```

## Run (VS Code terminal)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   |  source .venv/bin/activate  (Mac/Linux)
pip install -r requirements.txt
python main.py                  # trains, evaluates, saves bundle + reports
streamlit run app.py            # interactive scorer
```

## Data cleaning highlights
- 165 duplicate rows removed; 5 impossible ages (up to 144) removed.
- Employment length of 123 years set to missing (cannot exceed age − 14).
- `loan_percent_income` recomputed (338 rows were inconsistent).
- Missing `emp_length` (889) and `loan_int_rate` (3,094): flags added (missing
  emp_length defaults ~31% vs ~21.5%), then imputed — interest rate by grade median.
- Income/loan outliers (income up to 6M) capped at the train 99.5th percentile.
- Caps/medians are learned on **train only** to avoid leakage.

## Notes
- `loan_grade` and `loan_int_rate` are set by the lender and already encode
  its own risk view. Set `USE_LENDER_PRICING = False` in `config.py` to see
  performance without them.
- Cutoff is chosen on validation data with a cost matrix (FN = 5 × FP), then
  applied once to the test set.
