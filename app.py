import streamlit as st
import joblib
import pandas as pd
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Dark / Light Mode Toggle (Centered)
# -----------------------------
col1, col2, col3 = st.columns([3,2,3])
with col2:
    mode = st.toggle("🌙 Dark Mode")

# -----------------------------
# Animated Background + Responsive CSS
# -----------------------------
st.markdown(f"""
<style>

.stApp {{
    background: linear-gradient(-45deg,
        {'#0f2027, #203a43, #2c5364' if mode else '#667eea, #764ba2, #43cea2'}
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Segoe UI', sans-serif;
}}

@keyframes gradientBG {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

/* Center Glass Card */
.block-container {{
    background: {'rgba(0,0,0,0.45)' if mode else 'rgba(255,255,255,0.22)'};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 2rem;
    border-radius: 20px;
    max-width: 720px;
    margin: 40px auto;
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}}

h1 {{
    text-align: center;
    color: white;
    font-weight: 700;
}}

.desc {{
    text-align: center;
    color: #f1f1f1;
    font-size: 15px;
    margin-bottom: 20px;
}}

label {{
    color: white !important;
    font-weight: 600;
}}

div.stButton > button {{
    width: 100%;
    background: linear-gradient(to right, #43cea2, #185a9d);
    color: white;
    font-size: 17px;
    padding: 0.7em;
    border-radius: 14px;
    border: none;
    margin-top: 15px;
}}

div.stButton > button:hover {{
    transform: scale(1.04);
    box-shadow: 0 0 18px rgba(67,206,162,0.8);
}}

.stAlert {{
    border-radius: 14px;
    font-size: 16px;
    text-align: center;
}}

.reco {{
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 14px;
    margin-top: 15px;
    color: white;
}}

.footer {{
    text-align: center;
    color: #dddddd;
    font-size: 13px;
    margin-top: 25px;
}}

@media (max-width: 768px) {{
    .block-container {{
        max-width: 95%;
        margin: 20px auto;
    }}
}}

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
Predict whether a student will <b>Pass or Fail</b><br>
using <b>Machine Learning (Logistic Regression)</b><br>
with <b>Smart Recommendation System</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Inputs
# -----------------------------
study_hours = st.slider("📘 Study Hours per Day", 0.0, 10.0, step=0.1)
attendance = st.slider("📊 Attendance Percentage", 0.0, 100.0, step=1.0)

# -----------------------------
# Prediction + Recommendation
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

        # ✅ Recommendation (PASS)
        st.markdown("""
        <div class="reco">
        ✅ <b>Recommendations:</b>
        <ul>
            <li>Maintain consistent study schedule</li>
            <li>Revise daily & solve practice questions</li>
            <li>Keep attendance above 80%</li>
            <li>Focus on weak subjects to score higher</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("❌ STUDENT WILL FAIL")
        st.info(f"📉 Fail Probability: {(1-prob)*100:.2f}%")

        # ❌ Recommendation (FAIL)
        tips = []
        if study_hours < 3:
            tips.append("Increase study hours to at least 4–5 hours/day")
        if attendance < 75:
            tips.append("Improve attendance to minimum 75%")
        tips.append("Make a daily study timetable")
        tips.append("Attend doubt-solving sessions")
        tips.append("Reduce mobile/social media usage")

        st.markdown("""
        <div class="reco">
        ❌ <b>Recommendations to Improve:</b>
        <ul>
        """ + "".join([f"<li>{t}</li>" for t in tips]) + """
        </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit, Machine Learning & Recommendation System
</div>
""", unsafe_allow_html=True)
