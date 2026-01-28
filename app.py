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
# Dark / Light Mode Toggle (Centered)
# -----------------------------
col1, col2, col3 = st.columns([3,2,3])
with col2:
    mode = st.toggle("🌙 Dark Mode")

# -----------------------------
# Animated Background + CSS
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

.block-container {{
    background: {'rgba(0,0,0,0.45)' if mode else 'rgba(255,255,255,0.22)'};
    backdrop-filter: blur(16px);
    padding: 2rem;
    border-radius: 20px;
    max-width: 720px;
    margin: 40px auto;
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
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
    font-weight: 600;
}}

input {{
    border-radius: 12px !important;
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

div.stButton > button:hover {{
    transform: scale(1.04);
    box-shadow: 0 0 18px rgba(67,206,162,0.8);
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
Pass / Fail Prediction using <b>Machine Learning</b><br>
with <b>Smart Recommendation System</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# TEXTBOX INPUTS
# -----------------------------
study_hours = st.text_input("📘 Study Hours per Day", placeholder="Enter hours (e.g. 5)")
attendance = st.text_input("📊 Attendance Percentage", placeholder="Enter % (e.g. 82)")

# -----------------------------
# Prediction + Recommendation
# -----------------------------
if st.button("🔍 Predict Result"):

    try:
        study_hours = float(study_hours)
        attendance = float(attendance)

        if study_hours < 0 or attendance < 0 or attendance > 100:
            st.error("❌ Please enter valid values")
        else:
            df = pd.DataFrame([[study_hours, attendance]],
                              columns=["StudyHours", "Attendance"])

            scaled = scaler.transform(df)
            pred = model.predict(scaled)
            prob = model.predict_proba(scaled)[0][1]

            st.divider()

            if pred[0] == 1:
                st.success("🎉 STUDENT WILL PASS")
                st.info(f"📈 Pass Probability: {prob*100:.2f}%")

                st.markdown("""
                <div class="reco">
                ✅ <b>Recommendations:</b>
                <ul>
                    <li>Maintain regular study routine</li>
                    <li>Practice previous year questions</li>
                    <li>Keep attendance above 80%</li>
                    <li>Revise weak subjects</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("❌ STUDENT WILL FAIL")
                st.info(f"📉 Fail Probability: {(1-prob)*100:.2f}%")

                tips = []
                if study_hours < 4:
                    tips.append("Increase study hours to at least 4–5 hrs/day")
                if attendance < 75:
                    tips.append("Improve attendance above 75%")
                tips += [
                    "Create a daily timetable",
                    "Reduce mobile usage",
                    "Attend doubt-solving sessions"
                ]

                st.markdown("""
                <div class="reco">
                ❌ <b>Recommendations to Improve:</b>
                <ul>
                """ + "".join([f"<li>{t}</li>" for t in tips]) + """
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
