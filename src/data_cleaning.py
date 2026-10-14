"""
Data cleaning in two stages:
  Stage A  - rule-based fixes (no statistics learned) -> safe before the split.
  Stage B  - statistics learned on TRAIN only (caps, medians) -> no leakage.
"""
import numpy as np
import pandas as pd

from .config import CAT_COLS, CAT_LEVELS, TARGET


def clean_stage_a(df: pd.DataFrame):
    log = [f"Raw rows: {len(df)}"]
    df = df.copy()

    # 1. Tidy text columns; unknown levels -> dropped (home ownership -> OTHER)
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip().str.upper()
    df.loc[~df["person_home_ownership"].isin(CAT_LEVELS["person_home_ownership"]),
           "person_home_ownership"] = "OTHER"
    ok = pd.Series(True, index=df.index)
    for c in CAT_COLS:
        ok &= df[c].isin(CAT_LEVELS[c])
    log.append(f"Rows with unknown category levels removed: {(~ok).sum()}")
    df = df[ok]

    # 2. Duplicates
    n = df.duplicated().sum()
    df = df.drop_duplicates()
    log.append(f"Duplicate rows removed: {n}")

    # 3. Target sanity
    df = df[df[TARGET].isin([0, 1])]

    # 4. Impossible ages (max in raw data is 144)
    bad = (df["person_age"] < 18) | (df["person_age"] > 100)
    log.append(f"Impossible ages removed (<18 or >100): {bad.sum()}")
    df = df[~bad]

    # 5. Impossible employment length (cannot work longer than age - 14)
    bad = df["person_emp_length"] > (df["person_age"] - 14)
    df.loc[bad, "person_emp_length"] = np.nan
    log.append(f"Impossible emp_length set to NaN: {bad.sum()}")

    # 6. Credit history cannot be longer than adult life
    bad = df["cb_person_cred_hist_length"] > (df["person_age"] - 14)
    df.loc[bad, "cb_person_cred_hist_length"] = np.nan
    log.append(f"Impossible credit history set to NaN: {bad.sum()}")

    # 7. Non-positive money values
    for c in ["person_income", "loan_amnt"]:
        bad = df[c] <= 0
        if bad.any():
            df = df[~bad]
            log.append(f"Non-positive {c} removed: {bad.sum()}")

    # 8. Missing-value flags (missingness itself is predictive here:
    #    applicants with no emp_length default ~31% vs ~21.5%)
    df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
    df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
    log.append(f"Missing emp_length: {df['emp_length_missing'].sum()}, "
               f"missing int_rate: {df['int_rate_missing'].sum()}")

    # 9. Recompute loan_percent_income (338 raw rows are inconsistent)
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]

    log.append(f"Rows after stage A: {len(df)}  |  default rate: {df[TARGET].mean():.2%}")
    return df.reset_index(drop=True), log


class StageBCleaner:
    """Learn caps / medians on train only, apply to train, valid, test."""

    NUM_CAP = ["person_income", "loan_amnt", "person_age", "person_emp_length",
               "cb_person_cred_hist_length"]

    def __init__(self, upper_q: float = 0.995):
        self.upper_q = upper_q

    def fit(self, df: pd.DataFrame):
        self.caps = {c: df[c].quantile(self.upper_q) for c in self.NUM_CAP}
        self.medians = {c: df[c].clip(upper=self.caps[c]).median() for c in self.NUM_CAP}
        # interest rate: median within loan grade (rates are set by grade)
        self.rate_by_grade = df.groupby("loan_grade")["loan_int_rate"].median()
        self.rate_global = df["loan_int_rate"].median()
        return self

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        for c in self.NUM_CAP:
            df[c] = df[c].clip(upper=self.caps[c]).fillna(self.medians[c])
        fill = df["loan_grade"].map(self.rate_by_grade).fillna(self.rate_global)
        df["loan_int_rate"] = df["loan_int_rate"].fillna(fill)
        # capped income/loan -> recompute ratio
        df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
        return df
