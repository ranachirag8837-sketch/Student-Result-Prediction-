import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="🎓 Student Result Prediction AI",
    layout="wide"
)

# =============================
# Custom CSS (Borders Removed)
# =============================
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #4B0082;
        color: white;
    }

    /* Info Container (No Border) */
    .info-border-box {
        border: none !important; 
        border-radius: 25px;
        padding: 40px;
        background-color: rgba(255, 255, 255, 0.07);
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }

    /* Input Box Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border: none !important; 
        text-align: center;
        border-radius: 12px;
        height: 45px;
    }

    /* Centering labels */
    .stTextInput > label {
        display: flex;
        justify-content: center;
        color: #ffffff !important;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Predict Button */
    div.stButton {
        text-align: center;
        margin-top: 25px;
    }

    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 12px;
        padding: 0.6rem 2.5rem;
        border: none !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# Load Dataset & Training
# =============================
df = pd.DataFrame({
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 45, 50, 60, 65, 75, 80, 85, 90, 95],
    "ResultNumeric": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "TotalMarks": [25, 30, 38, 45, 55, 68, 75, 82, 88, 95]
})

X = df[["StudyHours", "Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logistic_model = LogisticRegression().fit(X_scaled, df["ResultNumeric"])
linear_model = LinearRegression().fit(X_scaled, df["TotalMarks"])

# =============================
# Main Layout
# =============================
col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    st.markdown('<div class="info-border-box">', unsafe_allow_html=True)
    
    st.markdown("""
        <h1 style="font-size: 3.5rem; font-weight: 800; color: white; margin-bottom: 0;">🎓 Student Result Prediction</h1>
        <p style="color: rgba(255,255,255,0.7); margin-bottom: 30px; font-size: 1.1rem;">
            🔹 Hybrid ML Model: Logistic & Linear Regression
        </p>
    """, unsafe_allow_html=True)

    study_hours_input = st.text_input("📘 Study Hours (per day)", value="8")
    attendance_input = st.text_input("📊 Attendance (%)", value="85")
    predict_clicked = st.button("🌟 Predict Result") 
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Prediction Logic
# =============================
if predict_clicked:
    try:
        sh = float(study_hours_input)
        at = float(attendance_input)
        
        # Prepare Input
        input_data = pd.DataFrame([[sh, at]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        # Predict
        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(input_scaled)[0]), 100.0)

        # Result Card HTML
        res_text = "PASS" if pass_prob >= 0.5 else "FAIL"
        res_color = "#4ade80" if pass_prob >= 0.5 else "#f87171"

        html_code = f"""
        <html>
        <head>
          <script src="https://cdn.tailwindcss.com"></script>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        </head>
        <body class="bg-transparent flex justify-center">
          <div class="max-w-xl w-full">
            <div class="bg-white/10 backdrop-blur-lg rounded-3xl p-8 text-center shadow-2xl">
              <h2 class="text-3xl font-bold mb-6 text-white">Prediction Result</h2>
              <div class="space-y-2 mb-6 text-white">
                <p class="text-xl">Pass Probability: <span class="font-bold text-blue-300">{pass_prob*100:.2f}%</span></p>
                <p class="text-xl">Predicted Marks: <span class="font-bold text-blue-300">{pred_marks:.2f} / 100</span></p>
              </div>
              <div class="text-[{res_color}] font-black text-5xl mb-6">🎉 RESULT: {res_text}</div>
              <script>
                if({str(pass_prob >= 0.5).lower()}) {{
                    confetti({{ particleCount: 150, spread: 70, origin: {{ y: 0.6 }} }});
                }}
              </script>
            </div>
          </div>
        </body>
        </html>
        """
        
        with col_mid:
            components.html(html_code, height=350)

            # --- GRAPH SECTION ---
            st.write("---")
            st.markdown("<h3 style='text-align: center;'>📊 Performance Analytics Graph</h3>", unsafe_allow_html=True)
            
            

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#4B0082') 
            ax.set_facecolor('#ffffff10')

            # Scatter plot of training data
            ax.scatter(df['StudyHours'], df['Attendance'], c=df['ResultNumeric'], cmap='RdYlGn', s=100, label='Historical Data', alpha=0.5)
            # User Data Point (The Star)
            ax.scatter(sh, at, color='cyan', marker='*', s=300, label='Your Prediction', edgecolors='white')

            ax.set_xlabel('Study Hours', color='white')
            ax.set_ylabel('Attendance (%)', color='white')
            ax.tick_params(colors='white')
            ax.legend()
            st.pyplot(fig)
            
    except ValueError:
        st.error("⚠️ Please enter valid numeric values.")

# Footer
st.markdown("<br><center><p style='color: white; opacity: 0.5;'>AI/ML Project | Clean Interface</p></center>", unsafe_allow_html=True)
