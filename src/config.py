"""Central configuration: paths, column names, constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"

SEED = 42
TARGET = "loan_status"          # 1 = default, 0 = repaid

# Categorical columns and their allowed levels (fixed => no train/test mismatch)
CAT_LEVELS = {
    "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "OTHER"],
    "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL",
                    "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}
CAT_COLS = list(CAT_LEVELS)

# Scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double odds
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20

# Business cost matrix used to choose the approval cutoff
COST_FN = 5.0   # approving someone who defaults (lose principal)
COST_FP = 1.0   # rejecting someone who would have repaid (lose margin)

# Set False to build a scorecard WITHOUT lender-assigned pricing
# (loan_grade / loan_int_rate already encode the lender's own risk view)
USE_LENDER_PRICING = True

# Explainable raw / ratio features allowed in the scorecard (LightGBM uses all)
SCORECARD_CANDIDATES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file",
    "cb_person_cred_hist_length", "emp_length_missing",
]
