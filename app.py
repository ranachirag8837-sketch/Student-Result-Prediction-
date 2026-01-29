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
# Custom CSS for Black Theme & Transparency
# =============================
st.markdown("""
<style>
    /* Global Black Background */
    .stApp {
        background-color: #4B0082;
        color: white;
    }

    /* Centering the inputs and labels */
    .stTextInput > label {
        display: flex;
        justify-content: center;
        color: #ffffff !important;
        font-weight: bold;
    }

    /* Style for Input Boxes (Semi-transparent) */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: black !important;
        border: 1px solid #444 !important;
        text-align: center;
        border-radius: 10px;
    }

    /* Center the Predict button */
    div.stButton {
        text-align: center;
        margin-top: 20px;
    }

    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Load Dataset & Training
# =============================
# Using dummy data if file is missing
df = pd.DataFrame({
    "StudyHours": [1,2,3,4,5,6,7,8],
    "Attendance": [45,50,55,60,70,80,90,95],
    "ResultNumeric": [0,0,0,1,1,1,1,1],
    "TotalMarks": [30,35,40,50,60,70,85,92]
})

X = df[["StudyHours", "Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logistic_model = LogisticRegression().fit(X_scaled, df["ResultNumeric"])
linear_model = LinearRegression().fit(X_scaled, df["TotalMarks"])

# =============================
# Centered Header Section
# =============================
st.markdown("""
    <div style="text-align: center; padding: 40px 0 10px 0;">
        <h1 style="font-size: 3rem; font-weight: 800; color: white;">🎓 Student Result Prediction</h1>
        <p style="color: #94a3b8; margin-bottom: 30px;">
            🔹 Hybrid ML Model: Logistic Regression & Linear Regression
        </p>
    </div>
""", unsafe_allow_html=True)

# =============================
# Centered Vertical Input Section
# =============================
# Creating columns to pinch the center for vertical stacking
col_left, col_mid, col_right = st.columns([1, 1.5, 1])

with col_mid:
    # Attendance is now at the bottom of Study Hours
    study_hours_input = st.text_input("📘 Study Hours (per day)", value="10")
    attendance_input = st.text_input("📊 Attendance (%)", value="100")
    
    predict_clicked = st.button("🌟 Predict Result")

# =============================
# Prediction Logic
# =============================
if predict_clicked:
    try:
        sh = float(study_hours_input)
        at = float(attendance_input)
        
        input_data = pd.DataFrame([[sh, at]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(input_scaled)[0]), 100.0)

        # HTML Result Card with Transparent Backdrop Blur
        html_code = f"""
        <html>
        <head>
          <script src="https://cdn.tailwindcss.com"></script>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        </head>
        <body class="bg-transparent flex justify-center">
          <div class="max-w-xl w-full mt-10">
            <div class="bg-white/10 backdrop-blur-lg rounded-3xl p-8 text-center border border-white/20 shadow-2xl">
              <h2 class="text-3xl font-bold mb-6 text-white">Result</h2>
              <div class="space-y-2 mb-6">
                <p class="text-xl text-gray-200">Pass Probability: <span class="font-bold text-blue-400">{pass_prob*100:.2f}%</span></p>
                <p class="text-xl text-gray-200">Predicted Marks: <span class="font-bold text-blue-400">{pred_marks:.2f} / 100</span></p>
              </div>
        """

        if pass_prob >= 0.5:
            html_code += f"""
              <div class="text-green-400 font-black text-3xl mb-6">🎉 RESULT: PASS</div>
              <script>
                confetti({{ particleCount: 150, spread: 70, origin: {{ y: 0.6 }} }});
              </script>
            """
        else:
            html_code += '<div class="text-red-400 font-black text-3xl mb-6">❌ RESULT: FAIL</div>'

        html_code += f"""
              <div class="bg-black/30 p-4 rounded-xl text-left border border-white/10">
                <h3 class="font-bold text-white mb-1">💡 Recommendation:</h3>
                <p class="text-gray-300">{"Keep up the excellent work!" if pred_marks > 70 else "Focus on consistency to improve marks."}</p>
              </div>
            </div>
          </div>
        </body>
        </html>
        """

        with col_mid:
            components.html(html_code, height=500)
            
    except ValueError:
        st.error("⚠️ Please enter valid numeric values.")

# =============================
# Footer
# =============================
st.markdown("<br><center><p style='color: #444;'>Built with ❤️ | Dark Mode Active</p></center>", unsafe_allow_html=True)


