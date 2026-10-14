"""Streamlit demo:  streamlit run app.py   (run `python main.py` first)"""
import joblib
import streamlit as st

from src.config import BUNDLE_PATH, CAT_LEVELS
from src.predict import score_applicant

st.set_page_config(page_title="Credit Risk Scorer", page_icon="💳", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


if not BUNDLE_PATH.exists():
    st.error("Model bundle not found. Run `python main.py` first.")
    st.stop()
bundle = load_bundle()

st.title("💳 Credit Risk Scorer")
st.caption("Probability of default (LightGBM + logistic regression) and scorecard points")

with st.sidebar:
    st.header("Applicant")
    age = st.number_input("Age", 18, 100, 28)
    income = st.number_input("Annual income", 4000, 500000, 55000, step=1000)
    home = st.selectbox("Home ownership", CAT_LEVELS["person_home_ownership"])
    emp_known = st.checkbox("Employment length known", True)
    emp = st.number_input("Employment length (years)", 0.0, 50.0, 4.0, step=0.5,
                          disabled=not emp_known)
    intent = st.selectbox("Loan purpose", CAT_LEVELS["loan_intent"])
    grade = st.selectbox("Loan grade", CAT_LEVELS["loan_grade"], index=1)
    amount = st.number_input("Loan amount", 500, 35000, 9000, step=500)
    rate_known = st.checkbox("Interest rate known", True)
    rate = st.number_input("Interest rate (%)", 5.0, 25.0, 11.0, step=0.1,
                           disabled=not rate_known)
    prior = st.selectbox("Previous default on file", CAT_LEVELS["cb_person_default_on_file"])
    hist = st.number_input("Credit history length (years)", 0, 40, 4)

if st.sidebar.button("Score applicant", type="primary"):
    raw = dict(person_age=age, person_income=income, person_home_ownership=home,
               person_emp_length=emp if emp_known else None, loan_intent=intent,
               loan_grade=grade, loan_amnt=amount,
               loan_int_rate=rate if rate_known else None,
               cb_person_default_on_file=prior, cb_person_cred_hist_length=hist)
    r = score_applicant(bundle, raw)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PD – LightGBM", f"{r['pd_lightgbm']:.1%}")
    c2.metric("PD – Logistic/Scorecard", f"{r['pd_logreg']:.1%}")
    c3.metric("Credit score", r["score"])
    c4.metric("Decision", r["decision"])
    st.progress(min(max((r["score"] - 350) / (680 - 350), 0.0), 1.0),
                text="Score position (350 = riskiest, 680 = safest)")
    st.caption(f"Approval rule: reject/review if LightGBM PD ≥ {r['cutoff_pd']:.1%} "
               "(cutoff chosen on validation data with FN cost = 5 × FP cost).")

    st.subheader("Why this score? (largest point losses)")
    st.dataframe(r["detail"].rename(columns={
        "points_lost_vs_best": "points lost vs best bin"}), hide_index=True)
else:
    st.info("Fill in the applicant on the left and click **Score applicant**.")
