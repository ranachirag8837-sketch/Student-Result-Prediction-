import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="🎓 Student Result Prediction",
    page_icon="🎓",
    layout="wide"
)

# =============================
# Load Dataset
# =============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    df = pd.DataFrame({
        "StudyHours": [1,2,3,4,5,6,7,8],
        "Attendance": [45,50,55,60,70,80,90,95],
        "ResultNumeric": [0,0,0,1,1,1,1,1],
        "TotalMarks": [30,35,40,50,60,70,85,92]
    })

required_columns = ["StudyHours", "Attendance", "ResultNumeric", "TotalMarks"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Missing required column: {col}")
        st.stop()

# =============================
# Model Training
# =============================
X = df[["StudyHours", "Attendance"]]
y_class = df["ResultNumeric"]
y_marks = df["TotalMarks"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logistic_model = LogisticRegression()
logistic_model.fit(X_scaled, y_class)

linear_model = LinearRegression()
linear_model.fit(X_scaled, y_marks)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
/* General Page */
body {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Center all content */
.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

/* Input Boxes */
.stTextInput>div>input {
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    border: 2px solid #ddd;
    transition: 0.3s;
}
.stTextInput>div>input:focus {
    border-color: #4CAF50;
    box-shadow: 0 0 10px rgba(76, 175, 80, 0.4);
}

/* Button */
.stButton>button {
    background: linear-gradient(to right, #4CAF50, #81C784);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 12px 25px;
    border-radius: 12px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
}

/* Result Text */
.result-pass {
    color: #2E7D32;
    font-weight: bold;
    font-size: 24px;
}
.result-fail {
    color: #C62828;
    font-weight: bold;
    font-size: 24px;
}

/* Recommendations */
.recommendation {
    background: #f0f0f0;
    border-radius: 15px;
    padding: 15px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown('<h1 style="text-align:center; font-size:32px; margin-bottom:15px;">🎓 Student Result Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px; color:#555;">Hybrid ML Model: Logistic Regression → Pass/Fail, Linear Regression → Marks Prediction</p>', unsafe_allow_html=True)

# =============================
# Inputs
# =============================
col1, col2 = st.columns(2)
with col1:
    study_hours_input = st.text_input("📘 Study Hours (per day)", value="")
with col2:
    attendance_input = st.text_input("📊 Attendance (%)", value="")

try:
    study_hours = float(study_hours_input) if study_hours_input.strip() != "" else None
except:
    study_hours = None

try:
    attendance = float(attendance_input) if attendance_input.strip() != "" else None
except:
    attendance = None

# =============================
# Prediction Button
# =============================
if st.button("🌟 Predict Result"):

    if study_hours is None or attendance is None:
        st.warning("⚠️ Please enter both Study Hours and Attendance before predicting.")
    else:
        input_data = pd.DataFrame([[study_hours, attendance]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = linear_model.predict(input_scaled)[0]

        # =============================
        # Result Card
        # =============================
        if pass_prob >= 0.5 and pred_marks >= 40:
            st.markdown(f'<div class="card result-pass">🎉 RESULT: PASS</div>', unsafe_allow_html=True)

            # Full-page confetti
            components.html("""
            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
            <script>
            function launchConfetti() {
                var duration = 4000;
                var animationEnd = Date.now() + duration;
                var defaults = { startVelocity: 40, spread: 360, ticks: 60, zIndex: 9999 };
                var interval = setInterval(function() {
                    var timeLeft = animationEnd - Date.now();
                    if (timeLeft <= 0) return clearInterval(interval);
                    var particleCount = 100 * (timeLeft / duration);
                    confetti(Object.assign({}, defaults, {
                        particleCount: particleCount,
                        origin: { x: Math.random(), y: Math.random() - 0.2 }
                    }));
                }, 250);
            }
            launchConfetti();
            </script>
            """, height=0)

        else:
            st.markdown(f'<div class="card result-fail">❌ RESULT: FAIL</div>', unsafe_allow_html=True)

        # =============================
        # Pass Probability & Marks
        # =============================
        st.markdown(f'<div class="card"><b>📈 Pass Probability:</b> {pass_prob*100:.2f}%<br><b>📝 Predicted Marks:</b> {pred_marks:.2f}/100</div>', unsafe_allow_html=True)

        # =============================
        # Recommendations
        # =============================
        rec_html = '<div class="card recommendation"><b>💡 Recommendations:</b><ul>'
        if pass_prob < 0.5 or pred_marks < 40:
            rec_html += """
            <li>Increase <strong>study hours</strong> per day</li>
            <li>Improve <strong>attendance</strong> in classes</li>
            <li>Focus on weak subjects or topics</li>
            <li>Practice previous exams and exercises</li>
            """
        elif pred_marks < 60:
            rec_html += """
            <li>Maintain or slightly increase study hours</li>
            <li>Keep attendance high</li>
            <li>Focus on revision and practice to improve marks</li>
            """
        else:
            rec_html += """
            <li>Excellent performance! Keep consistent study habits 🎉</li>
            """
        rec_html += '</ul></div>'
        st.markdown(rec_html, unsafe_allow_html=True)

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • Modern CSS Design")
