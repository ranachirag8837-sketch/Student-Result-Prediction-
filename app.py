import streamlit as st
import joblib
import os
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# 🎨 CUSTOM CSS + BACKGROUND
# -----------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    font-family: 'Segoe UI', sans-serif;
}

/* Main card */
.block-container {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(14px);
    padding: 2.5rem;
    border-radius: 18px;
    max-width: 650px;
    margin-top: 40px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Title */
h1 {
    text-align: center;
    color: #ffffff;
    font-weight: 700;
}

/* Description */
.desc {
    text-align: center;
    color: #f1f1f1;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Labels */
label {
    color: #ffffff !important;
    font-weight: 600;
}

/* Button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(to right, #43cea2, #185a9d);
    color: white;
    font-size: 18px;
    padding: 0.7em;
    border-radius: 12px;
    border: none;
    margin-top: 15px;
    transition: 0.3s ease;
}

div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(67, 206, 162, 0.7);
}

/* Alerts */
.stAlert {
    border-radius: 14px;
    font-size: 18px;
    text-align: center;
}

/* Footer */
.footer {
    text-align: center;
    color: #e0e0e0;
    font-size: 13px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model/logistic_model.pkl"
SCALER_PATH = "model/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Result Prediction System")
st.markdown("""
<div class="desc">
Predict whether a student will <b>Pass or Fail</b> using  
<b>Machine Learning (Logistic Regression)</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Inputs
# -----------------------------
study_hours = st.slider("📘 Study Hours per Day", 0.0, 10.0, step=0.1)
attendance = st.slider("📊 Attendance Percentage", 0.0, 100.0, step=1.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Result"):
    data = pd.DataFrame([[study_hours, attendance]],
                        columns=["StudyHours", "Attendance"])

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.success("🎉 STUDENT WILL PASS")
        st.info(f"📈 Pass Probability: {probability*100:.2f}%")
    else:
        st.error("❌ STUDENT WILL FAIL")
        st.info(f"📉 Fail Probability: {(1-probability)*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
