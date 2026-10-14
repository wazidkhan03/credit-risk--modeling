"""Score ONE applicant with the saved bundle (used by main.py and app.py)."""
import numpy as np
import pandas as pd

from .config import CAT_COLS
from .features import engineer
from .models import pd_from_score, predict_lgbm, score_from_logit


def prepare_applicant(raw: dict) -> pd.DataFrame:
    d = pd.DataFrame([raw])
    for c in CAT_COLS:
        d[c] = d[c].astype(str).str.strip().str.upper()
    adult = d["person_age"] - 14
    d.loc[d["person_emp_length"] > adult, "person_emp_length"] = np.nan
    d.loc[d["cb_person_cred_hist_length"] > adult, "cb_person_cred_hist_length"] = np.nan
    d["emp_length_missing"] = d["person_emp_length"].isna().astype(int)
    d["int_rate_missing"] = d["loan_int_rate"].isna().astype(int)
    d["loan_percent_income"] = d["loan_amnt"] / d["person_income"]
    return d


def score_applicant(bundle: dict, raw: dict) -> dict:
    feat = engineer(bundle["cleaner"].transform(prepare_applicant(raw)))
    sel, lr, binner = bundle["selected"], bundle["logreg"], bundle["binner"]

    pd_gbm = float(predict_lgbm(bundle["gbm"], feat[bundle["lgb_columns"]])[0])
    W = binner.transform(feat, sel)
    pd_lr = float(lr.predict_proba(W)[0, 1])
    score = float(score_from_logit(lr.decision_function(W))[0])

    # Per-feature points + "reason codes" (where the applicant loses most points)
    table = bundle["scorecard_table"]
    k, icpt = len(sel), lr.intercept_[0]
    from .models import scorecard_params
    offset, factor = scorecard_params()
    rows = []
    for c, beta in zip(sel, lr.coef_[0]):
        pts = -(beta * W[c].iloc[0] + icpt / k) * factor + offset / k
        best = table.loc[table.feature == c, "points"].max()
        v = feat[c].iloc[0]
        v = f"{v:.3g}" if isinstance(v, (int, float, np.number)) else str(v)
        rows.append(dict(feature=c, value=v, points=round(pts, 1),
                         points_lost_vs_best=round(best - pts, 1)))
    detail = pd.DataFrame(rows).sort_values("points_lost_vs_best", ascending=False)

    cutoff = bundle["cutoff_pd"]
    return dict(pd_lightgbm=pd_gbm, pd_logreg=pd_lr, score=round(score),
                pd_from_score=float(pd_from_score(score)),
                decision="REJECT / REVIEW" if pd_gbm >= cutoff else "APPROVE",
                cutoff_pd=cutoff, detail=detail)
