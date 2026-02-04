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
    page_title=" Student Result Prediction AI",
    layout="wide"
) 

# =============================
# 2. Custom CSS
# =============================
st.markdown("""
<style>
    .stApp {
        background-color: #4B0082;
        color: white;
    }

    .info-border-box {
        border: none !important; 
        border-radius: 25px;
        padding: 40px;
        background-color: rgba(255, 255, 255, 0.08);
        margin-top: 20px;
        margin-bottom: 20px;
        text-align: center;
    }

    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: black !important;
        border: none !important; 
        text-align: center;
        border-radius: 12px;
        height: 45px;
    }

    .stTextInput > label {
        display: flex;
        justify-content: center;
        color: #ffffff !important;
        font-weight: bold;
        margin-bottom: 10px;
    }

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
        width: 100%;
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
        <h1 style="font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 0;"> Student Result Prediction</h1>
        <p style="color: rgba(255,255,255,0.7); margin-bottom: 30px; font-size: 1rem;">
             Hybrid ML Model: Logistic & Linear Regression
        </p>
    """, unsafe_allow_html=True)

    study_hours_input = st.text_input(" Study Hours (per day)", value="8")
    attendance_input = st.text_input(" Attendance (%)", value="85")
    predict_clicked = st.button(" Predict Result") 
    
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

        res_text = "PASS" if pass_prob >= 0.5 else "FAIL"
        res_color = "#4ade80" if pass_prob >= 0.5 else "#f87171"
        rec_icon = "" if pass_prob >= 0.8 else "" if pass_prob >= 0.5 else ""
        rec_text = ("Excellent! Keep it up." if pass_prob >= 0.8 else 
                    "Safe zone, but can improve." if pass_prob >= 0.5 else 
                    "Warning! Need more effort.")

        html_code = f"""
        <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 20px; text-align: center; color: white; font-family: sans-serif;">
            <h2 style="margin-bottom: 10px;">Prediction Result</h2>
            <p>Pass Probability: <span style="color: #93c5fd; font-weight: bold;">{pass_prob*100:.1f}%</span></p>
            <p>Estimated Marks: <span style="color: #93c5fd; font-weight: bold;">{pred_marks:.1f} / 100</span></p>
            <h1 style="color: {res_color}; font-size: 3rem; margin: 10px 0;">{res_text}</h1>
            <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 10px; text-align: left; border-left: 4px solid #3b82f6;">
                <small><b>{rec_icon} AI Recommendation:</b></small><br>
                <small style="font-style: italic; color: #e5e7eb;">{rec_text}</small>
            </div>
        </div>
        """
        
        with col_mid:
            components.html(html_code, height=280)

            # --- Advanced Features Section ---
            st.markdown("""
            <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 25px; margin-top: 20px; border-left: 5px solid #3b82f6;">
                <h3 style="color: white; margin-bottom: 15px; font-size: 1.2rem;">Advanced Features & Recommendations</h3>
                <ul style="color: rgba(255,255,255,0.9); list-style-type: none; padding-left: 0; font-size: 0.9rem; line-height: 1.6;">
                    <li><b>1. Personalized Study Plan:</b> Generate schedule based on weak areas.</li>
                    <li><b>2. Topic-wise Analysis:</b> Detailed performance by subject.</li>
                    <li><b>3. Peer Comparison:</b> View anonymous comparative stats.</li>
                    <li><b>4. Goal Setting & Tracking:</b> Monitor achievement progress.</li>
                    <li><b>5. Tutor Connect:</b> Request AI or human tutor assistance.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if st.button("Request Feature ➔"):
                st.toast("Feature request sent! ")

            st.write("---")
            st.markdown("<h3 style='text-align: center;'> Study Hours vs Attendance Analysis</h3>", unsafe_allow_html=True)
            
            # Matplotlib Graph
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#4B0082') 
            ax.set_facecolor('#ffffff')
            colors = ['#4ade80' if r == 1 else '#f87171' for r in df['ResultNumeric']]
            ax.bar(df['StudyHours'], df['Attendance'], color=colors, alpha=0.5, width=0.6)
            ax.bar(sh, at, color='cyan', width=0.4, edgecolor='black', linewidth=2, label='Your Input')
            ax.set_xlabel('Study Hours', color='white')
            ax.set_ylabel('Attendance (%)', color='white')
            ax.tick_params(colors='white')
            st.pyplot(fig)
            
    except ValueError:
        st.error(" Please enter valid numbers.")

st.markdown("<br><center><p style='color: white; opacity: 0.5;'>Predictor v2.3 | AI Analytics Dashboard</p></center>", unsafe_allow_html=True)
