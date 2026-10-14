"""Metrics, business cutoff, and all plots."""
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from .config import COST_FN, COST_FP, REPORT_DIR
from .features import to_lgb_frame

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ks_stat(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))


def metrics_row(name, y, p, interp):
    auc = roc_auc_score(y, p)
    return dict(Model=name, AUC=auc, Gini=2 * auc - 1, KS=ks_stat(y, p),
                AvgPrecision=average_precision_score(y, p),
                Brier=brier_score_loss(y, np.clip(p, 0, 1)), Interpretability=interp)


def best_cutoff(y, p):
    """Cutoff minimising COST_FN*FN + COST_FP*FP."""
    grid = np.linspace(0.02, 0.9, 177)
    costs = []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int)).ravel()
        costs.append(COST_FN * fn + COST_FP * fp)
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def strategy_table(y, p):
    """Approval rate vs bad rate: how lenders actually set cut-offs."""
    rows = []
    for ap in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        t = np.quantile(p, ap)
        approved = p <= t
        rows.append(dict(approval_rate=ap, max_PD_accepted=round(t, 3),
                         bad_rate_in_approved=round(y[approved].mean(), 4),
                         defaults_caught=round(1 - y[approved].sum() / y.sum(), 3)))
    return pd.DataFrame(rows)


def plot_all(y, probs: dict, score, y_score, tag="test"):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for n, p in probs.items():
        f, t, _ = roc_curve(y, p)
        ax[0, 0].plot(f, t, label=f"{n} (AUC {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0, 1].plot(rc, pr, label=f"{n} (AP {average_precision_score(y, p):.3f})")
        fp, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[0, 2].plot(mp, fp, marker="o", label=n)
        o = np.argsort(p)
        cg = np.cumsum(1 - y.values[o]) / (1 - y).sum()
        cb = np.cumsum(y.values[o]) / y.sum()
        ax[1, 0].plot(np.linspace(0, 1, len(o)), cb - cg, label=f"{n} KS {ks_stat(y, p):.3f}")
    ax[0, 0].plot([0, 1], [0, 1], "k--"); ax[0, 0].set_title("ROC"); ax[0, 0].legend()
    ax[0, 1].set_title("Precision-Recall"); ax[0, 1].legend()
    ax[0, 1].axhline(y.mean(), color="k", ls="--", lw=.8)
    ax[0, 2].plot([0, 1], [0, 1], "k--"); ax[0, 2].set_title("Calibration")
    ax[0, 2].set_xlabel("Predicted PD"); ax[0, 2].set_ylabel("Observed default rate")
    ax[0, 2].legend()
    ax[1, 0].set_title("KS curve (bad% - good% cumulative)"); ax[1, 0].legend()
    ax[1, 1].hist(score[y_score == 0], 40, alpha=.6, density=True, label="Repaid")
    ax[1, 1].hist(score[y_score == 1], 40, alpha=.6, density=True, label="Default")
    ax[1, 1].set_title("Scorecard score distribution"); ax[1, 1].legend()
    bands = pd.qcut(score, 10, duplicates="drop")
    br = pd.DataFrame({"b": bands, "y": y_score.values}).groupby("b", observed=True)["y"].mean()
    ax[1, 2].bar(range(len(br)), br.values)
    ax[1, 2].set_title("Default rate by score decile (0 = riskiest)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"evaluation_{tag}.png", dpi=130)
    plt.close()


def plot_iv(iv):
    s = iv.sort_values()
    plt.figure(figsize=(7, 6))
    plt.barh(s.index, s.values)
    for x in (0.02, 0.1, 0.3):
        plt.axvline(x, color="k", ls=":", lw=.8)
    plt.title("Information Value (IV)"); plt.tight_layout()
    plt.savefig(REPORT_DIR / "information_value.png", dpi=130); plt.close()


def shap_report(gbm, X, n=2000, seed=42):
    smp = to_lgb_frame(X.sample(min(n, len(X)), random_state=seed))
    sv = shap.TreeExplainer(gbm).shap_values(smp)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    shap.summary_plot(sv, smp, show=False)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "shap_summary.png", dpi=130); plt.close()
    shap.summary_plot(sv, smp, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(REPORT_DIR / "shap_importance.png", dpi=130); plt.close()
