import streamlit as st
import pandas as pd
import numpy as np
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

.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    text-align: center;
}

.stButton > button {
    background-color: #3b82f6 !important;
    color: white !important;
    border-radius: 12px;
    padding: 0.6rem 2.5rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 3. Dataset & Training
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

logistic_model = LogisticRegression().fit(X_scaled, df["ResultNumeric"])
linear_model = LinearRegression().fit(X_scaled, df["TotalMarks"])

# =============================
# 4. Layout
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-border-box">', unsafe_allow_html=True)

    st.markdown("""
    <h1>🎓 Student Result Prediction</h1>
    <p>Hybrid ML Model (Logistic + Linear)</p>
    """, unsafe_allow_html=True)

    sh = st.text_input("📘 Study Hours", "8")
    at = st.text_input("📊 Attendance (%)", "85")
    predict = st.button("🌟 Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 5. Prediction + Charts (NO GRAPH)
# =============================
if predict:
    try:
        sh = float(sh)
        at = float(at)

        input_df = pd.DataFrame([[sh, at]], columns=["StudyHours","Attendance"])
        input_scaled = scaler.transform(input_df)

        pass_prob = logistic_model.predict_proba(input_scaled)[0][1]
        marks = min(float(linear_model.predict(input_scaled)[0]), 100)

        result = "PASS" if pass_prob >= 0.5 else "FAIL"
        color = "#4ade80" if result=="PASS" else "#f87171"

        # ================= Result Card =================
        html = f"""
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
        <div style="background:rgba(255,255,255,0.12);
                    padding:30px;
                    border-radius:25px;
                    text-align:center;">
            <h2>Prediction Result</h2>
            <h1 style="color:{color};">{result}</h1>
            <p>Probability: {pass_prob*100:.1f}%</p>
            <p>Estimated Marks: {marks:.1f}/100</p>
        </div>
        <script>
        if({str(pass_prob>=0.5).lower()}) {{
            confetti({{particleCount:120, spread:70}});
        }}
        </script>
        """
        components.html(html, height=300)

        # ================= CHART SECTION =================
        st.markdown("### 📊 Performance Chart")

        st.markdown("**Pass Probability**")
        st.progress(int(pass_prob*100))

        st.markdown("**Estimated Marks**")
        st.progress(int(marks))

    except:
        st.error("⚠️ Please enter valid numeric values")

# Footer
st.markdown(
"<center style='opacity:0.6;'>Predictor v2.3 | AI Dashboard</center>",
unsafe_allow_html=True
)
