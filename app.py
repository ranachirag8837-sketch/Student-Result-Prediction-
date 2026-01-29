import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="🎓 Student Result Prediction",
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
# User Inputs
# =============================
st.markdown("""
<style>
/* TailwindCSS CDN */
@import url('https://cdn.jsdelivr.net/npm/tailwindcss@3.3.2/dist/tailwind.min.css');

/* Custom Streamlit overrides */
body { background-color: #f0f2f6; }
.stButton>button { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="flex justify-center my-8">
  <div class="bg-white shadow-2xl rounded-2xl p-8 w-full max-w-2xl">
    <h1 class="text-3xl font-extrabold text-center mb-6">🎓 Student Result Prediction</h1>
    <p class="text-gray-700 text-center mb-6">
        🔹 Hybrid ML Model: Logistic Regression → Pass/Fail, Linear Regression → Marks Prediction
    </p>
""", unsafe_allow_html=True)

# Inputs in styled Tailwind cards
col1, col2 = st.columns([1,1])
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
# Prediction
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
        # Full-page Tailwind HTML
        # =============================
        html_code = f"""
        <html>
        <head>
          <script src="https://cdn.tailwindcss.com"></script>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        </head>
        <body class="bg-gray-100 flex flex-col items-center p-6">
          <div class="max-w-2xl w-full space-y-6">

            <div class="bg-white rounded-2xl shadow-2xl p-6 text-center animate-fadeIn">
              <h2 class="text-2xl font-bold mb-4">Result</h2>
              <p class="text-lg mb-2">Pass Probability: <span class="font-semibold">{pass_prob*100:.2f}%</span></p>
              <p class="text-lg mb-4">Predicted Marks: <span class="font-semibold">{pred_marks:.2f} / 100</span></p>
        """

        if pass_prob >= 0.5 and pred_marks >= 40:
            html_code += """
              <div class="text-green-600 font-bold text-2xl mb-4">🎉 RESULT: PASS</div>
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
            """
        else:
            html_code += '<div class="text-red-600 font-bold text-2xl mb-4">❌ RESULT: FAIL</div>'

        # Recommendations
        html_code += '<div class="bg-gray-50 p-4 rounded-lg shadow-inner">'
        html_code += '<h3 class="font-bold text-lg mb-2">💡 Recommendations:</h3>'
        if pass_prob < 0.5 or pred_marks < 40:
            html_code += """
            <ul class="list-disc list-inside text-left text-gray-700">
              <li>Increase <strong>study hours</strong> per day</li>
              <li>Improve <strong>attendance</strong> in classes</li>
              <li>Focus on weak subjects or topics</li>
              <li>Practice previous exams and exercises</li>
            </ul>
            """
        elif pred_marks < 60:
            html_code += """
            <ul class="list-disc list-inside text-left text-gray-700">
              <li>Maintain or slightly increase study hours</li>
              <li>Keep attendance high</li>
              <li>Focus on revision and practice to improve marks</li>
            </ul>
            """
        else:
            html_code += """
            <ul class="list-disc list-inside text-left text-gray-700">
              <li>Excellent performance! Keep consistent study habits 🎉</li>
            </ul>
            """
        html_code += "</div></div></body></html>"

        components.html(html_code, height=700, scrolling=True)

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit • Modern TailwindCSS design")

