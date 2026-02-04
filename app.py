import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="🎓 Student Result Prediction AI",
    page_icon="🎓",
    layout="wide"
)

# =============================
# CSS (Smaller Size & Center Alignment)
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}

/* Centering the input labels */
.stTextInput label {
    display: flex;
    justify-content: center;
    font-size: 14px !important;
    font-weight: bold;
}

/* Smaller Input Box */
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
    height: 35px; /* Reduced height */
    text-align: center;
    font-size: 14px;
    width: 60% !important; /* Smaller width */
    margin: auto;
}

/* Smaller Centered Button */
.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 38px; /* Reduced height */
    width: 120px; /* Reduced width */
    font-size: 14px;
    font-weight: bold;
    border: none;
    margin-top: 10px;
}

.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Dataset & Models
# =============================
df = pd.DataFrame({
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [40,45,50,60,65,75,80,85,90,95],
    "Result": [0,0,0,0,1,1,1,1,1,1],
    "Marks": [25,30,38,45,55,68,75,82,88,95]
})

X = df[["StudyHours", "Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression().fit(X_scaled, df["Result"])
lin_model = LinearRegression().fit(X_scaled, df["Marks"])

# =============================
# UI INPUT (CENTERED)
# =============================
col1, col2, col3 = st.columns([1,1.5,1]) # Adjusted column ratio for better focus

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom:0;'>Student Result Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px; opacity:0.8;'>Hybrid ML Model (Pass / Fail + Marks)</p>", unsafe_allow_html=True)

    sh = st.text_input("Study Hours", "8")
    at = st.text_input("Attendance %", "85")
    predict = st.button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction Logic
# =============================
if predict:
    sh_val = float(sh)
    at_val = float(at)

    inp = scaler.transform([[sh_val, at_val]])
    pass_prob = log_model.predict_proba(inp)[0][1]
    marks = min(lin_model.predict(inp)[0], 100)

    color = "#22c55e" if pass_prob >= 0.5 else "#ef4444"
    level = "Excellent" if pass_prob >= 0.8 else "Good" if pass_prob >= 0.6 else "Needs Improvement"
    advice = "Maintain consistency." if pass_prob >= 0.8 else "Focus on weak areas." if pass_prob >= 0.6 else "Increase effort immediately."
    progress = int(pass_prob * 100)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        components.html(f"""
        <div style="background:rgba(255,255,255,0.18); padding:20px; border-radius:20px; text-align:center; color:white; font-family:sans-serif;">
            <h3 style="margin:0;">Result</h3>
            <p style="margin:5px 0;">Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{color}; margin:10px 0;">{'PASS' if pass_prob >= 0.5 else 'FAIL'}</h1>
        </div>
        """, height=180)

    # Advanced Features (Full Width)
    st.markdown("---")
    components.html(f"""
    <div style="width:100%; background:linear-gradient(135deg,#6a11cb,#2575fc); border-radius:25px; padding:30px; color:white; font-family:sans-serif;">
        <h2>Advanced Recommendations</h2>
        <p>Status: <b style="color:{color}">{level}</b></p>
        <div style="background:rgba(255,255,255,0.2); border-radius:10px; height:15px; width:100%; margin-bottom:20px;">
            <div style="background:{color}; width:{progress}%; height:100%; border-radius:10px;"></div>
        </div>
        <p><b>AI Advice:</b> {advice}</p>
    </div>
    """, height=250)

# =============================
# Footer
# =============================
st.markdown(
    "<br><center style='opacity:0.6; font-size:12px;'>Predictor v2.6 | AI Analytics Dashboard</center>",
    unsafe_allow_html=True
)
