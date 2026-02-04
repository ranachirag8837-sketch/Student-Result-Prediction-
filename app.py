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
# 2. Custom CSS
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}
.info-border-box {
    border-radius: 25px;
    padding: 40px;
    background-color: rgba(255,255,255,0.08);
    text-align: center;
}
.stTextInput input {
    background-color: white !important;
    color: black !important;
    text-align: center;
    border-radius: 12px;
    height: 45px;
}
.stTextInput label {
    display: flex;
    justify-content: center;
    font-weight: bold;
}
.stButton button {
    background-color: #3b82f6;
    color: white;
    border-radius: 12px;
    padding: 0.6rem 2.5rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 3. Dataset & Model Training
# =============================
df = pd.DataFrame({
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [40,45,50,60,65,75,80,85,90,95],
    "ResultNumeric": [0,0,0,0,1,1,1,1,1,1],
    "TotalMarks": [25,30,38,45,55,68,75,82,88,95]
})

X = df[["StudyHours", "Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logistic_model = LogisticRegression()
logistic_model.fit(X_scaled, df["ResultNumeric"])

linear_model = LinearRegression()
linear_model.fit(X_scaled, df["TotalMarks"])

# =============================
# 4. Layout
# =============================
_, col_mid, _ = st.columns([1,2,1])

with col_mid:
    st.markdown('<div class="info-border-box">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style="font-size:3rem;font-weight:800;">🎓 Student Result Prediction</h1>
        <p style="opacity:0.7;">Hybrid ML Model (Logistic + Linear Regression)</p>
    """, unsafe_allow_html=True)

    study_hours_input = st.text_input("📘 Study Hours (per day)")
    attendance_input = st.text_input("📊 Attendance (%)")
    predict_clicked = st.button("🌟 Predict Result")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 5. Prediction + Advanced Recommendation
# =============================
if predict_clicked:
    try:
        sh = float(study_hours_input)
        at = float(attendance_input)

        input_df = pd.DataFrame([[sh, at]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_df)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        pred_marks = min(float(linear_model.predict(input_scaled)[0]), 100)

        # =============================
        # ADVANCED AI RECOMMENDATION LOGIC
        # =============================
        recommendations = []

        if sh < 3:
            recommendations.append("📘 Increase study time to at least **3–4 hours/day** for concept clarity.")
        elif sh < 6:
            recommendations.append("📘 Study hours are decent, but **1 extra hour daily** can boost marks.")
        else:
            recommendations.append("📘 Excellent study consistency. Focus on **revision & mock tests**.")

        if at < 60:
            recommendations.append("📊 Attendance is critically low. Aim for **75%+ attendance** immediately.")
        elif at < 75:
            recommendations.append("📊 Improve attendance slightly to strengthen internal performance.")
        else:
            recommendations.append("📊 Attendance is strong. Maintain regular class participation.")

        if pred_marks < 40:
            recommendations.append("❗ High risk zone: Revise fundamentals and seek teacher guidance.")
        elif pred_marks < 60:
            recommendations.append("⚠️ Average performance: Practice previous year questions weekly.")
        elif pred_marks < 80:
            recommendations.append("📈 Good score range: Focus on weak topics for distinction.")
        else:
            recommendations.append("🏆 Excellent marks predicted: Start preparing for competitive exams.")

        final_recommendation = "<br>".join(recommendations)

        res_text = "PASS" if pass_prob >= 0.5 else "FAIL"
        res_color = "#4ade80" if res_text == "PASS" else "#f87171"

        # =============================
        # Result UI
        # =============================
        html_code = f"""
        <html>
        <head>
          <script src="https://cdn.tailwindcss.com"></script>
          <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        </head>
        <body class="flex justify-center">
          <div class="max-w-xl w-full bg-white/10 backdrop-blur-lg rounded-3xl p-8 text-white text-center">
            <h2 class="text-3xl font-bold mb-4">Prediction Result</h2>
            <p class="text-xl">Pass Probability: <b class="text-blue-300">{pass_prob*100:.1f}%</b></p>
            <p class="text-xl mb-4">Estimated Marks: <b class="text-blue-300">{pred_marks:.1f}/100</b></p>
            <div class="text-5xl font-black mb-4" style="color:{res_color};">{res_text}</div>

            <div class="bg-black/30 p-4 rounded-xl text-left border-l-4 border-blue-400">
              <p class="font-bold mb-2">🤖 AI Personalized Recommendation:</p>
              <p class="text-sm leading-relaxed">{final_recommendation}</p>
            </div>

            <script>
              if({str(pass_prob >= 0.5).lower()}) {{
                confetti({{ particleCount: 160, spread: 70, origin: {{ y: 0.6 }} }});
              }}
            </script>
          </div>
        </body>
        </html>
        """

        with col_mid:
            components.html(html_code, height=480)

            st.write("---")
            st.markdown("<h3 style='text-align:center;'>📊 Study Hours vs Attendance Analysis</h3>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10,5))
            colors = ['#4ade80' if r == 1 else '#f87171' for r in df['ResultNumeric']]
            ax.bar(df['StudyHours'], df['Attendance'], color=colors, alpha=0.6)
            ax.bar(sh, at, color='cyan', width=0.4)

            ax.set_xlabel("Study Hours")
            ax.set_ylabel("Attendance (%)")
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            st.pyplot(fig)

    except ValueError:
        st.error("⚠️ Please enter valid numeric values only.")

# =============================
# Footer
# =============================
st.markdown(
    "<br><center style='opacity:0.6;'>Predictor v3.0 | Advanced AI Recommendation Engine</center>",
    unsafe_allow_html=True
)
