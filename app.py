import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Dark / Light Mode Toggle
# -----------------------------
col1, col2, col3 = st.columns([3,2,3])
with col2:
    mode = st.toggle("🌙 Dark Mode")

# -----------------------------
# CSS + Animated Background
# -----------------------------
st.markdown(f"""
<style>

.stApp {{
    background: linear-gradient(-45deg,
        {'#1f1c2c, #928DAB' if mode else '#667eea, #764ba2'}
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

.block-container {{
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(18px);
    padding: 2rem;
    border-radius: 22px;
    max-width: 720px;
    margin: 40px auto;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
}}

h1 {{
    text-align: center;
    color: white;
}}

.desc {{
    text-align: center;
    color: #f1f1f1;
}}

label {{
    color: white !important;
}}

.result-card {{
    background: rgba(120,130,200,0.45);
    padding: 18px;
    border-radius: 16px;
    margin-top: 15px;
    font-size: 18px;
    font-weight: 600;
    color: #00ffcc;
}}

.prob-card {{
    background: rgba(140,150,220,0.45);
    padding: 14px;
    border-radius: 14px;
    margin-top: 12px;
    color: #e6f0ff;
}}

.fail {{
    color: #ff6b6b;
}}

.reco {{
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 14px;
    margin-top: 15px;
    color: white;
}}

div.stButton > button {{
    width: 100%;
    background: linear-gradient(to right, #43cea2, #185a9d);
    color: white;
    font-size: 17px;
    padding: 0.7em;
    border-radius: 14px;
    border: none;
}}

.footer {{
    text-align: center;
    color: #dddddd;
    font-size: 13px;
    margin-top: 25px;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Result Prediction System")
st.markdown("""
<div class="desc">
Pass / Fail Prediction<br>
<b>with Smart Recommendation System</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Inputs (Textbox)
# -----------------------------
study_hours = st.text_input("📘 Study Hours per Day", placeholder="e.g. 5")
attendance = st.text_input("📊 Attendance Percentage", placeholder="e.g. 80")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Result"):
    try:
        study_hours = float(study_hours)
        attendance = float(attendance)

        if study_hours < 0 or attendance < 0 or attendance > 100:
            st.error("❌ Enter valid values")
        else:
            df = pd.DataFrame([[study_hours, attendance]],
                              columns=["StudyHours", "Attendance"])
            scaled = scaler.transform(df)
            pred = model.predict(scaled)
            prob = model.predict_proba(scaled)[0][1]

            st.divider()

            if pred[0] == 1:
                st.markdown(f"""
                <div class="result-card">
                🎉 STUDENT WILL PASS
                </div>

                <div class="prob-card">
                📈 Pass Probability: <b>{prob*100:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="reco">
                ✅ <b>Recommendations:</b>
                <ul>
                    <li>Maintain regular study routine</li>
                    <li>Practice mock tests</li>
                    <li>Keep attendance above 80%</li>
                    <li>Focus on weak subjects</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-card fail">
                ❌ STUDENT WILL FAIL
                </div>

                <div class="prob-card">
                📉 Fail Probability: <b>{(1-prob)*100:.2f}%</b>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="reco">
                ❌ <b>Recommendations to Improve:</b>
                <ul>
                    <li>Increase study hours to 4–5 hrs/day</li>
                    <li>Improve attendance above 75%</li>
                    <li>Create daily study timetable</li>
                    <li>Avoid distractions</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

    except ValueError:
        st.error("⚠️ Please enter numeric values only")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
