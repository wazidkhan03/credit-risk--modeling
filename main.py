"""
Run the full pipeline:   python main.py
Steps: load -> clean -> split -> clean(B) -> features -> IV/WOE -> models
       -> evaluate -> SHAP -> save bundle + reports
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (BUNDLE_PATH, DATA_PATH, MODEL_DIR, REPORT_DIR, SEED,
                        TARGET, USE_LENDER_PRICING)
from src.data_cleaning import StageBCleaner, clean_stage_a
from src.evaluate import (best_cutoff, metrics_row, plot_all, plot_iv,
                          shap_report, strategy_table)
from src.features import engineer
from src.models import (build_scorecard, fit_lgbm, fit_logreg, pd_from_score,
                        predict_lgbm, score_from_logit, select_features)
from src.predict import score_applicant
from src.woe import WOEBinner

pd.set_option("display.width", 160)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# 1 ─ LOAD ------------------------------------------------------------
raw = pd.read_csv(DATA_PATH)
print(f"Raw data: {raw.shape}")
print("Missing values:\n", raw.isna().sum()[raw.isna().sum() > 0], "\n")

# 2 ─ CLEAN (stage A: rule-based) -------------------------------------
df, log = clean_stage_a(raw)
print("── Cleaning log ──\n" + "\n".join(log))
(REPORT_DIR / "cleaning_log.txt").write_text("\n".join(log))

# 3 ─ SPLIT (stratified 60/15/25) — BEFORE learning any statistic -----
y, X = df[TARGET], df.drop(columns=TARGET)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.20,
                                          stratify=y_tr, random_state=SEED)
print(f"\nTrain {X_tr.shape} | Valid {X_va.shape} | Test {X_te.shape}")

# 4 ─ CLEAN (stage B: fitted on train only) + FEATURES ----------------
cleaner = StageBCleaner().fit(X_tr)
X_tr, X_va, X_te = [engineer(cleaner.transform(d)) for d in (X_tr, X_va, X_te)]
assert X_tr.isna().sum().sum() == 0, "NaNs left after cleaning"
print(f"Features after engineering: {X_tr.shape[1]}")

# 5 ─ WOE / IV + selection --------------------------------------------
binner = WOEBinner().fit(X_tr, y_tr)
iv = binner.iv()
iv.to_csv(REPORT_DIR / "information_value.csv")
plot_iv(iv)
print("\n── Information Value ──\n", iv.round(3).to_string())
sel0 = select_features(binner, X_tr)

# 6 ─ MODEL A: logistic regression on WOE ------------------------------
logreg, sel = fit_logreg(binner, X_tr, y_tr, sel0)
print(f"\nScorecard features ({len(sel)}): {sel}")
W_te = binner.transform(X_te, sel)
p_lr = logreg.predict_proba(W_te)[:, 1]

# 7 ─ MODEL B: LightGBM with random-search tuning ---------------------
print("\n── Tuning LightGBM ──")
gbm, tuning = fit_lgbm(X_tr, y_tr, X_va, y_va, n_trials=12)
tuning.to_csv(REPORT_DIR / "lgbm_tuning.csv", index=False)
p_gbm = predict_lgbm(gbm, X_te)

# 8 ─ MODEL C: scorecard ----------------------------------------------
sc_table = build_scorecard(logreg, binner, sel)
sc_table.to_csv(REPORT_DIR / "scorecard_points_table.csv", index=False)
score_te = score_from_logit(logreg.decision_function(W_te))
p_score = pd_from_score(score_te)
print(f"\nScore range on test: {score_te.min():.0f} – {score_te.max():.0f}")

# 9 ─ EVALUATE ---------------------------------------------------------
res = pd.DataFrame([
    metrics_row("Logistic Regression (WOE)", y_te, p_lr, "High"),
    metrics_row("LightGBM", y_te, p_gbm, "Low (needs SHAP)"),
    metrics_row("Scorecard (points)", y_te, p_score, "Very high"),
]).round(4)
res.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
print("\n══ MODEL COMPARISON (test set) ══\n", res.to_string(index=False))

# cutoff chosen on VALIDATION data, then applied to test (no peeking)
cut, _ = best_cutoff(y_va, predict_lgbm(gbm, X_va))
pred = (p_gbm >= cut).astype(int)
from sklearn.metrics import classification_report, confusion_matrix
print(f"\nCost-optimal cutoff (FN costs 5x FP) from validation: PD >= {cut:.3f}")
print("Confusion matrix on test [[TN FP],[FN TP]]:\n", confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, target_names=["Repaid", "Default"]))

strat = strategy_table(y_te, p_gbm)
strat.to_csv(REPORT_DIR / "strategy_table.csv", index=False)
print("── Approval strategy (LightGBM) ──\n", strat.to_string(index=False))

bands = pd.qcut(score_te, 10, duplicates="drop")
band_tbl = (pd.DataFrame({"band": bands, "y": y_te.values})
            .groupby("band", observed=True)["y"].agg(["count", "mean"])
            .rename(columns={"mean": "default_rate"}))
band_tbl.to_csv(REPORT_DIR / "score_bands.csv")
print("\n── Default rate by score decile (low score = risky) ──\n", band_tbl.round(3))

plot_all(y_te, {"LogReg": p_lr, "LightGBM": p_gbm}, score_te, y_te)

# 10 ─ EXPLAIN ---------------------------------------------------------
shap_report(gbm, X_te)

# 11 ─ SAVE ------------------------------------------------------------
bundle = dict(cleaner=cleaner, binner=binner, logreg=logreg, gbm=gbm,
              selected=sel, lgb_columns=list(X_tr.columns),
              scorecard_table=sc_table, cutoff_pd=cut,
              use_lender_pricing=USE_LENDER_PRICING)
joblib.dump(bundle, BUNDLE_PATH)

example = dict(person_age=24, person_income=42000, person_home_ownership="RENT",
               person_emp_length=2, loan_intent="PERSONAL", loan_grade="C",
               loan_amnt=12000, loan_int_rate=13.5,
               cb_person_default_on_file="N", cb_person_cred_hist_length=3)
r = score_applicant(bundle, example)
print("\n── Example applicant ──")
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "detail"})
print(r["detail"].head(5).to_string(index=False))
print(f"\nSaved bundle -> {BUNDLE_PATH}\nReports -> {REPORT_DIR}")
