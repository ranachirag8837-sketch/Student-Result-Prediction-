import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction (Hybrid)",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load models & scaler
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_DIR = os.path.join(ROOT_DIR, "model")

LOGISTIC_PATH = os.path.join(MODEL_DIR, "hybrid_logistic.pkl")
LINEAR_PATH = os.path.join(MODEL_DIR, "hybrid_linear.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "hybrid_scaler.pkl")

if not (os.path.exists(LOGISTIC_PATH) and os.path.exists(LINEAR_PATH)):
    st.error("❌ Hybrid models not found. Train the hybrid model first.")
    st.stop()

hybrid_logistic = joblib.load(LOGISTIC_PATH)
hybrid_linear = joblib.load(LINEAR_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Title & description
# -----------------------------
st.title("🎓 Student Result Prediction System (Hybrid Model)")
st.markdown(
    """
    This system uses a **Hybrid Machine Learning Model**:
    - **Logistic Regression** → Pass / Fail Probability  
    - **Linear Regression** → Expected Marks  

    Final decision is based on **both probability & marks**.
    """
)

st.divider()

# -----------------------------
# User inputs
# -----------------------------
study_hours = st.slider(
    "📘 Study Hours (per day)",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

attendance = st.slider(
    "📊 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Result"):
    input_df = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["StudyHours", "Attendance"]
    )

    input_scaled = scaler.transform(input_df)

    # Logistic prediction
    pass_prob = hybrid_logistic.predict_proba(input_scaled)[0][1]

    # Linear prediction
    predicted_marks = hybrid_linear.predict(input_scaled)[0]

    # Hybrid decision
    if pass_prob >= 0.5 and predicted_marks >= 40:
        final_result = "PASS"
        st.success("🎉 **PASS**")
    else:
        final_result = "FAIL"
        st.error("❌ **FAIL**")

    st.divider()

    st.info(f"📈 **Pass Probability:** {pass_prob*100:.2f}%")
    st.info(f"📝 **Predicted Marks:** {predicted_marks:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hybrid Machine Learning Model")
