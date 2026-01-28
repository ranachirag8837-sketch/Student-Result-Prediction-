import streamlit as st
import joblib
import pandas as pd
import os
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Function: Load Background Image
# -----------------------------
def add_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# 🎨 Glass UI CSS
# -----------------------------
st.markdown("""
<style>

/* Glass card */
.block-container {
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 2.5rem;
    border-radius: 20px;
    max-width: 650px;
    margin-top: 40px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
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

/* Hover */
div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px rgba(67, 206, 162, 0.8);
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
    color: #ffffff;
    font-size: 13px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
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
    df = pd.DataFrame([[study_hours, attendance]],
                      columns=["StudyHours", "Attendance"])

    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)[0][1]

    st.divider()

    if pred[0] == 1:
        st.success("🎉 STUDENT WILL PASS")
        st.info(f"📈 Pass Probability: {prob*100:.2f}%")
    else:
        st.error("❌ STUDENT WILL FAIL")
        st.info(f"📉 Fail Probability: {(1-prob)*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)

