"""Model training: WOE logistic regression, tuned LightGBM, scorecard."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import BASE_ODDS, BASE_SCORE, PDO, SCORECARD_CANDIDATES, SEED
from .features import to_lgb_frame


# ---------------------------------------------------------------- features
def select_features(binner, X_tr, iv_min=0.02, corr_max=0.6):
    """Scorecard features: simple, explainable candidates only (no interactions)."""
    iv = binner.iv()
    iv = iv[[c for c in iv.index if c in SCORECARD_CANDIDATES]]
    sel = iv[iv >= iv_min].index.tolist()
    W = binner.transform(X_tr, sel)
    corr = W.corr().abs()
    drop = set()
    for i, a in enumerate(sel):
        for b in sel[i + 1:]:
            if a not in drop and b not in drop and corr.loc[a, b] > corr_max:
                drop.add(b if iv[b] < iv[a] else a)      # keep the higher-IV one
    return [c for c in sel if c not in drop]


def fit_logreg(binner, X_tr, y_tr, sel):
    """Drop any feature whose WOE coefficient is positive (breaks scorecard logic)."""
    sel = list(sel)
    iv = binner.iv()
    while True:
        lr = LogisticRegression(max_iter=2000).fit(binner.transform(X_tr, sel), y_tr)
        pos = [c for c, b in zip(sel, lr.coef_[0]) if b > 0]
        if not pos:
            return lr, sel
        sel.remove(min(pos, key=lambda c: iv[c]))


# ---------------------------------------------------------------- LightGBM
def fit_lgbm(X_tr, y_tr, X_va, y_va, n_trials=12, verbose=True):
    rng = np.random.default_rng(SEED)
    Xt, Xv = to_lgb_frame(X_tr), to_lgb_frame(X_va)
    space = dict(num_leaves=[7, 15, 31, 63], learning_rate=[0.02, 0.05, 0.1],
                 min_child_samples=[20, 50, 100], feature_fraction=[0.7, 0.9, 1.0],
                 lambda_l2=[0.0, 5.0, 20.0])
    best, best_auc, history = None, -1, []
    for t in range(n_trials):
        p = {k: v[rng.integers(len(v))] for k, v in space.items()}
        p = {k: (int(v) if k in ("num_leaves", "min_child_samples") else float(v))
             for k, v in p.items()}
        p.update(objective="binary", metric="auc", bagging_fraction=0.8,
                 bagging_freq=1, seed=SEED, verbose=-1)
        dtr = lgb.Dataset(Xt, y_tr)                 # fresh Dataset per trial
        dva = lgb.Dataset(Xv, y_va, reference=dtr)
        m = lgb.train(p, dtr, 1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        auc = float(m.best_score["valid_0"]["auc"])
        history.append({**{k: p[k] for k in space}, "iters": m.best_iteration,
                        "valid_auc": round(auc, 4)})
        if verbose:
            print(f"  trial {t + 1:>2}/{n_trials}  valid AUC={auc:.4f}  {history[-1]}")
        if auc > best_auc:
            best, best_auc = m, auc
    return best, pd.DataFrame(history).sort_values("valid_auc", ascending=False)


def predict_lgbm(model, X):
    return model.predict(to_lgb_frame(X), num_iteration=model.best_iteration)


# ---------------------------------------------------------------- scorecard
def scorecard_params():
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return offset, factor


def build_scorecard(logreg, binner, sel):
    """Points per bin:  -(beta*WOE + intercept/k)*factor + offset/k"""
    offset, factor = scorecard_params()
    k, icpt = len(sel), logreg.intercept_[0]
    rows = []
    for c, beta in zip(sel, logreg.coef_[0]):
        for _, r in binner.tables[c].iterrows():
            pts = -(beta * r["woe"] + icpt / k) * factor + offset / k
            rows.append(dict(feature=c, bin=r["bin"], label=r["label"],
                             count=int(r["count"]), bad_rate=round(r["bad_rate"], 4),
                             woe=round(r["woe"], 3), points=round(pts, 1)))
    return pd.DataFrame(rows)


def score_from_logit(logit_bad):
    offset, factor = scorecard_params()
    return offset - factor * logit_bad


def pd_from_score(score):
    offset, factor = scorecard_params()
    return 1 / (1 + np.exp(-(offset - score) / factor))
