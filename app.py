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
# 🎨 ADVANCED CUSTOM CSS
# -----------------------------
st.markdown(
    """
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #667eea, #764ba2);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Center main container */
    .block-container {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        padding: 2.5rem;
        border-radius: 18px;
        max-width: 650px;
        margin-top: 40px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }

    /* Title */
    h1 {
        text-align: center;
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Description text */
    .desc {
        text-align: center;
        color: #f1f1f1;
        font-size: 16px;
        margin-bottom: 25px;
    }

    /* Slider labels */
    label {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(to right, #43cea2, #185a9d);
        color: white;
        font-size: 18px;
        padding: 0.7em;
        border-radius: 12px;
        border: none;
        margin-top: 15px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(67, 206, 162, 0.7);
    }

    /* Result cards */
    .stAlert {
        border-radius: 14px;
        font-size: 18px;
        text-align: center;
    }

    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.3);
        margin: 30px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #e0e0e0;
        font-size: 13px;
        margin-top: 20px;
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
    <div class="desc">
        Predict whether a student will <b>Pass or Fail</b> using
        <b>Machine Learning (Logistic Regression)</b><br>
        based on Study Hours & Attendance
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# User inputs
# -----------------------------
study_hours = st.slider(
    "📘 Study Hours per Day",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

attendance = st.slider(
    "📊 Attendance Percentage",
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
        st.info(f"📈 **Pass Probability:** {probability*100:.2f}%")
    else:
        st.error("❌ **STUDENT WILL FAIL**")
        st.info(f"📉 **Fail Probability:** {(1 - probability)*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
