import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load model & scaler
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(ROOT_DIR, "model", "logistic_model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "model", "scaler.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Title & description
# -----------------------------
st.title("🎓 Student Result Prediction System")
st.markdown(
    """
    This system predicts whether a student will **Pass or Fail**
    using **Logistic Regression** based on:
    - Study Hours
    - Attendance
    """
)

st.divider()

# -----------------------------
# User inputs (LIVE)
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
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.success(f"🎉 **PASS**")
    else:
        st.error(f"❌ **FAIL**")

    st.info(f"📈 **Pass Probability:** {probability*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Logistic Regression")
