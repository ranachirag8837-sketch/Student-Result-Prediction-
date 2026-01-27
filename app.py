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
# 🔥 CUSTOM CSS STYLE
# -----------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(to right, #fdfbfb, #ebedee);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    h1 {
        color: #2c3e50;
        text-align: center;
    }

    /* Sub text */
    p {
        font-size: 16px;
        color: #34495e;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 2em;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.03);
    }

    /* Sliders */
    .stSlider > label {
        font-weight: bold;
        color: #2c3e50;
    }

    /* Result boxes */
    .stAlert {
        border-radius: 10px;
        font-size: 18px;
    }

    /* Footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
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
    - 📘 Study Hours  
    - 📊 Attendance  
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
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.success("🎉 **STUDENT WILL PASS**")
    else:
        st.error("❌ **STUDENT WILL FAIL**")

    st.info(f"📈 **Pass Probability:** {probability*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Logistic Regression")
