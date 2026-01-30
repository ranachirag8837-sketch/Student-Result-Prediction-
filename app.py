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
# 2. Custom CSS (Purple Theme & No Borders)
# =============================
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #4B0082;
        color: white;
    }

    /* Main Container (Border Removed) */
    .main-box {
        border: none !important; 
        border-radius: 30px;
        padding: 50px;
        background-color: rgba(255, 255, 255, 0.08); 
        margin-bottom: 25px;
        text-align: center;
    }

    /* Input Box Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #000 !important;
        border: none !important; 
        border-radius: 12px;
        text-align: center;
        height: 50px;
    }

    /* Predict Button */
    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 15px;
        padding: 0.8rem 3.5rem;
        border: none !important;
        font-weight: 800;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #2563eb !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 3. AI/ML Dataset & Model Training
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

# AI Models
logistic_model = LogisticRegression().fit(X_scaled, df["ResultNumeric"])
linear_model = LinearRegression().fit(X_scaled, df["TotalMarks"])

# =============================
# 4. User Interface Layout
# =============================
col_l, col_m, col_r = st.columns([1, 2, 1])

with col_m:
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🎓 Student Prediction AI</h1>
        <p style="opacity: 0.8; font-size: 1.2rem; margin-bottom: 40px;">
            AI-Driven Academic Performance Analysis
        </p>
    """, unsafe_allow_html=True)

    sh_input = st.text_input("📘 Study Hours (per day)", value="8")
    at_input = st.text_input("📊 Attendance (%)", value="85")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🌟 PREDICT NOW")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 5. Prediction Logic & Visualization
# =============================
if predict_clicked:
    try:
        sh = float(sh_input)
        at = float(at_input)
        
        # Scaling Input for AI
        user_data = pd.DataFrame([[sh, at]], columns=["StudyHours", "Attendance"])
        user_scaled = scaler.transform(user_data)

        # Inference
        pass_prob = logistic_model.predict_proba(user_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(user_scaled)[0]), 100.0)

        # Result Display (No Border)
        res_text = "PASS" if pass_prob >= 0.5 else "FAIL"
        res_color = "#4ade80" if pass_prob >= 0.5 else "#f87171"

        st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); padding: 30px; border-radius: 25px; text-align: center; margin-top: 20px;">
                <h2 style="color: {res_color}; font-size: 50px; font-weight: 900;">{res_text}</h2>
                <p style="font-size: 1.3rem;">Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
                <p style="font-size: 1.3rem;">Estimated Marks: <b>{pred_marks:.1f} / 100</b></p>
            </div>
        """, unsafe_allow_html=True)

        # Confetti Effect
        if pass_prob >= 0.5:
            components.html("""
                <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
                <script>confetti({particleCount: 150, spread: 70, origin: {y: 0.6}});</script>
            """, height=0)

        # --- AI GRAPH SECTION ---
        st.write("---")
        st.markdown("<h3 style='text-align: center;'>📊 Machine Learning Visual Analysis</h3>", unsafe_allow_html=True)
        
        

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#4B0082') 
        ax.set_facecolor('#310055')

        # Dataset Plotting
        ax.scatter(df['StudyHours'], df['Attendance'], c=df['ResultNumeric'], 
                   cmap='RdYlGn', s=100, edgecolors='white', alpha=0.6, label='Dataset')
        
        # Current User Prediction Point (Star Marker)
        ax.scatter(sh, at, color='cyan', marker='*', s=400, label='Your Data', edgecolors='white', linewidth=2)

        ax.set_xlabel('Study Hours', color='white')
        ax.set_ylabel('Attendance (%)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#4B0082', labelcolor='white')
        ax.grid(True, linestyle='--', alpha=0.2)

        st.pyplot(fig)

    except ValueError:
        st.error("⚠️ Please enter valid numeric values.")

st.markdown("<br><center><p style='opacity: 0.5;'>Built with ❤️ for AI/ML Project</p></center>", unsafe_allow_html=True)
