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
# Custom CSS for Centering
# =============================
st.markdown("""
<style>
    /* Center the button */
    div.stButton {
        text-align: center;
    }
    
    /* Center text inputs labels and fields */
    .stTextInput > label {
        display: flex;
        justify-content: center;
    }
    
    /* Global background */
    .stApp {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Load Dataset
# =============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    # Dummy data for demonstration
    df = pd.DataFrame({
        "StudyHours": [1,2,3,4,5,6,7,8],
        "Attendance": [45,50,55,60,70,80,90,95],
        "ResultNumeric": [0,0,0,1,1,1,1,1],
        "TotalMarks": [30,35,40,50,60,70,85,92]
    })

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
# Centered Header Section
# =============================
st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 2.5rem; font-weight: 800;">🎓 Student Result Prediction</h1>
        <p style="color: #4a5568; margin-bottom: 30px;">
            🔹 Hybrid ML Model: Logistic Regression → Pass/Fail, Linear Regression → Marks Prediction
        </p>
    </div>
""", unsafe_allow_html=True)

# =============================
# Centered Input Section
# =============================
# Creating 3 columns; the middle one (index 1) contains the content to keep it centered
col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    inner_col1, inner_col2 = st.columns(2)
    with inner_col1:
        study_hours_input = st.text_input("📘 Study Hours (per day)", value="10")
    with inner_col2:
        attendance_input = st.text_input("📊 Attendance (%)", value="100")
    
    predict_clicked = st.button("🌟 Predict Result")

# =============================
# Prediction Logic
# =============================
try:
    study_hours = float(study_hours_input) if study_hours_input.strip() != "" else None
    attendance = float(attendance_input) if attendance_input.strip() != "" else None
except:
    study_hours, attendance = None, None

if predict_clicked:
    if study_hours is None or attendance is None:
        st.warning("⚠️ Please enter both Study Hours and Attendance.")
    else:
        input_data = pd.DataFrame([[study_hours, attendance]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(input_scaled)[0]), 100.0)

        # HTML Result Card
        html_code = f"""
        <html>
        <head>
          <script src="https://cdn.tailwindcss.com"></script>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
          <style>
            @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            .animate-fadeIn {{ animation: fadeIn 0.5s ease-out forwards; }}
          </style>
        </head>
        <body class="bg-transparent flex justify-center">
          <div class="max-w-xl w-full mt-4 animate-fadeIn">
            <div class="bg-white rounded-3xl shadow-2xl p-8 text-center border border-gray-100">
              <h2 class="text-3xl font-bold mb-6 text-gray-800">Result</h2>
              <div class="space-y-2 mb-6">
                <p class="text-xl">Pass Probability: <span class="font-bold text-blue-600">{pass_prob*100:.2f}%</span></p>
                <p class="text-xl">Predicted Marks: <span class="font-bold text-blue-600">{pred_marks:.2f} / 100</span></p>
              </div>
        """

        if pass_prob >= 0.5 and pred_marks >= 40:
            html_code += f"""
              <div class="text-green-500 font-black text-3xl mb-6">🎉 RESULT: PASS</div>
              <script>
                confetti({{ particleCount: 150, spread: 70, origin: {{ y: 0.6 }} }});
              </script>
            """
        else:
            html_code += '<div class="text-red-500 font-black text-3xl mb-6">❌ RESULT: FAIL</div>'

        # Recommendations section
        html_code += '<div class="bg-gray-50 p-6 rounded-2xl text-left border border-gray-200">'
        html_code += '<h3 class="font-bold text-lg mb-3 flex items-center">💡 Recommendations:</h3>'
        
        if pred_marks >= 80:
            rec = "Excellent performance! Keep consistent study habits 🚀"
        elif pred_marks >= 40:
            rec = "Good job! Increase revision time to aim for higher marks."
        else:
            rec = "Focus on weak subjects and increase daily study hours significantly."
            
        html_code += f'<p class="text-gray-700 leading-relaxed">• {rec}</p>'
        html_code += "</div></div></div></body></html>"

        with col_mid:
            components.html(html_code, height=600)

# =============================
# Footer
# =============================
st.markdown("<br><hr><center><p style='color: gray;'>Built with ❤️ using Streamlit • Modern TailwindCSS design</p></center>", unsafe_allow_html=True)
