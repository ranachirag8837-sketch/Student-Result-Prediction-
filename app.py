
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components

# =============================
# 1. Page Configuration
# =============================
st.set_page_config(
    page_title="🎓 Student Result Prediction AI",
    layout="wide"
) 

# =============================
# 2. Custom CSS (No Borders, Clean UI)
# =============================
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #4B0082;
        color: white;
    }

    /* Main Container (Border Removed) */
    .info-border-box {
        border: none !important; 
        border-radius: 25px;
        padding: 40px;
        background-color: rgba(255, 255, 255, 0.08);
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

    /* Predict Button Styling */
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
# 3. Load Dataset & Training
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
# 4. Main Layout
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
# 5. Prediction Logic & Visualization
# =============================
if predict_clicked:
    try:
        sh = float(study_hours_input)
        at = float(attendance_input)

        input_data = pd.DataFrame([[sh, at]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(input_scaled)[0]), 100.0)

        if pass_prob >= 0.8:
            rec_text = "Excellent performance! Keep up the consistency and focus on advanced topics."
            rec_icon = "🚀"
        elif pass_prob >= 0.5:
            rec_text = "You are in the safe zone, but consider increasing study time by 1-2 hours to improve marks."
            rec_icon = "📈"
        else:
            rec_text = "Warning! You need to increase both attendance and study hours immediately to pass."
            rec_icon = "⚠️"

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
              <h2 class="text-3xl font-bold mb-4 text-white">Prediction Result</h2>
              <div class="space-y-2 mb-6 text-white">
                <p class="text-xl">Pass Probability: <span class="font-bold text-blue-300">{pass_prob*100:.1f}%</span></p>
                <p class="text-xl">Estimated Marks: <span class="font-bold text-blue-300">{pred_marks:.1f} / 100</span></p>
              </div>
              <div class="text-[{res_color}] font-black text-5xl mb-4">{res_text}</div>
              
              <div class="mt-4 p-4 bg-black/30 rounded-xl text-left border-l-4 border-blue-400">
                <p class="text-white text-sm"><b>{rec_icon} AI Recommendation:</b></p>
                <p class="text-gray-200 text-sm italic">{rec_text}</p>
              </div>

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
            components.html(html_code, height=450)

            st.write("---")
            st.markdown("<h3 style='text-align: center;'>📊 Study Hours vs Attendance Analysis</h3>", unsafe_allow_html=True)
            
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#4B0082') 
            ax.set_facecolor('#fff')

            colors = ['#4ade80' if r == 1 else '#f87171' for r in df['ResultNumeric']]
            
            ax.bar(df['StudyHours'], df['Attendance'], color=colors, alpha=0.6, label='Historical Data', width=0.6)
            
            ax.bar(sh, at, color='cyan', label='Your Input', width=0.4, edgecolor='white', linewidth=2)

            ax.set_xlabel('Study Hours', color='white', fontsize=12)
            ax.set_ylabel('Attendance (%)', color='white', fontsize=12)
            ax.tick_params(colors='white')
            
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='#4ade80', lw=4, alpha=0.6),
                            Line2D([0], [0], color='#f87171', lw=4, alpha=0.6),
                            Line2D([0], [0], color='cyan', lw=4)]
            ax.legend(custom_lines, ['Pass Path', 'Fail Path', 'Your Input'], 
                      facecolor='#4B0082', labelcolor='white')
            
            ax.grid(axis='y', linestyle='--', alpha=0.2)
            
            st.pyplot(fig)
            
    except ValueError:
        st.error("⚠️ Please enter valid numeric values.")

st.markdown("<br><center><p style='color: white; opacity: 0.5;'>Predictor v2.3 | AI Analytics Dashboard</p></center>", unsafe_allow_html=True)



