"""Feature engineering (row-level, so it is safe to apply to any split)."""
import numpy as np
import pandas as pd

from .config import CAT_COLS, CAT_LEVELS, USE_LENDER_PRICING

GRADE_NUM = {g: i + 1 for i, g in enumerate("ABCDEFG")}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    adult_years = (d["person_age"] - 17).clip(lower=1)

    # Burden / affordability
    d["annual_interest_cost"] = d["loan_amnt"] * d["loan_int_rate"] / 100
    d["interest_to_income"] = d["annual_interest_cost"] / d["person_income"]
    d["high_burden_flag"] = (d["loan_percent_income"] > 0.30).astype(int)

    # Stability / history
    d["emp_to_adult_years"] = d["person_emp_length"] / adult_years
    d["hist_to_adult_years"] = d["cb_person_cred_hist_length"] / adult_years
    d["short_history_flag"] = (d["cb_person_cred_hist_length"] <= 3).astype(int)

    # Scale transforms
    d["log_income"] = np.log1p(d["person_income"])
    d["log_loan"] = np.log1p(d["loan_amnt"])

    # Ordinal grade + interactions
    d["grade_num"] = d["loan_grade"].map(GRADE_NUM)
    d["prior_default_flag"] = (d["cb_person_default_on_file"] == "Y").astype(int)
    d["grade_x_burden"] = d["grade_num"] * d["loan_percent_income"]
    d["prior_default_x_burden"] = d["prior_default_flag"] * d["loan_percent_income"]

    if not USE_LENDER_PRICING:
        d = d.drop(columns=["loan_grade", "loan_int_rate", "grade_num",
                            "annual_interest_cost", "interest_to_income",
                            "grade_x_burden"])
    return d


def to_lgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals -> pandas 'category' with FIXED levels (for LightGBM)."""
    d = df.copy()
    for c in CAT_COLS:
        if c in d.columns:
            d[c] = pd.Categorical(d[c], categories=CAT_LEVELS[c])
    return d
