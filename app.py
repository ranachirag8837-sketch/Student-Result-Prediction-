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
# Custom CSS for Clean UI (Borders Removed)
# =============================
st.markdown("""
<style>
    /* Global Background (Deep Purple) */
    .stApp {
        background-color: #4B0082;
        color: white;
    }

    /* Info Container (Border Removed as per Image) */
    .info-border-box {
        border: none !important; 
        border-radius: 30px;
        padding: 50px;
        background-color: rgba(255, 255, 255, 0.07); /* Light overlay like the image */
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Centering labels */
    .stTextInput > label {
        display: flex;
        justify-content: center;
        color: #ffffff !important;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Input Box Styling (No Border) */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border: none !important; 
        outline: none !important;
        text-align: center;
        border-radius: 15px;
        height: 50px;
        box-shadow: none !important;
    }

    /* Predict Button Styling */
    div.stButton {
        text-align: center;
        margin-top: 30px;
    }

    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 15px;
        padding: 0.7rem 3rem;
        border: none !important;
        font-weight: bold;
        transition: 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Dataset & Model Training
# =============================
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
# Main UI Layout
# =============================
col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    # કન્ટેનર શરૂઆત (No Border)
    st.markdown('<div class="info-border-box">', unsafe_allow_html=True)
    
    st.markdown("""
        <h1 style="font-size: 3.2rem; font-weight: 800; color: white; margin-bottom: 10px;">🎓 Student Prediction</h1>
        <p style="color: rgba(255,255,255,0.7); margin-bottom: 35px; font-size: 1.1rem;">
            Fill in the details below to predict performance.
        </p>
    """, unsafe_allow_html=True)

    # Input Fields
    study_hours_input = st.text_input("📘 Study Hours", value="8")
    attendance_input = st.text_input("📊 Attendance (%)", value="85")
    predict_clicked = st.button("🌟 Predict Now") 
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Logic & Results
# =============================
if predict_clicked:
    try:
        sh = float(study_hours_input)
        at = float(attendance_input)
        
        input_data = pd.DataFrame([[sh, at]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(input_scaled)[0]), 100.0)

        # Result HTML (Tailwind borders also removed)
        html_code = f"""
        <html>
        <head>
          <script src="https://cdn.tailwindcss.com"></script>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        </head>
        <body class="bg-transparent flex justify-center">
          <div class="max-w-xl w-full">
            <div class="bg-white/10 backdrop-blur-xl rounded-3xl p-8 text-center shadow-2xl">
              <h2 class="text-2xl font-bold mb-4 text-white opacity-90">Prediction Summary</h2>
              <div class="grid grid-cols-2 gap-4 mb-6">
                <div class="bg-white/5 p-4 rounded-2xl">
                    <p class="text-sm text-gray-300">Probability</p>
                    <p class="text-2xl font-bold text-blue-300">{pass_prob*100:.1f}%</p>
                </div>
                <div class="bg-white/5 p-4 rounded-2xl">
                    <p class="text-sm text-gray-300">Est. Marks</p>
                    <p class="text-2xl font-bold text-blue-300">{pred_marks:.1f}</p>
                </div>
              </div>
        """

        if pass_prob >= 0.5:
            html_code += f"""
              <div class="text-green-400 font-black text-4xl mb-4">🏆 PASS</div>
              <script>
                confetti({{ particleCount: 150, spread: 70, origin: {{ y: 0.5 }} }});
              </script>
            """
        else:
            html_code += '<div class="text-red-400 font-black text-4xl mb-4">⚠️ FAIL</div>'

        html_code += """
            </div>
          </div>
        </body>
        </html>
        """

        with col_mid:
            components.html(html_code, height=400)
            
    except ValueError:
        st.error("❌ Please enter numeric values only.")

# Footer
st.markdown("<br><center><p style='color: white; opacity: 0.5;'>Predictor v2.0 | Clean Interface</p></center>", unsafe_allow_html=True)
