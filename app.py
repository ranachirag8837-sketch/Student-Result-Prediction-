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
# CSS (CENTERING LABELS & BUTTON)
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}

/* Center all labels for text inputs */
.stTextInput label {
    display: flex;
    justify-content: center;
    font-size: 18px !important;
    font-weight: bold !important;
}

/* Input Card Box */
.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
}

/* Text Input Box Styling */
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    height: 50px;
    text-align: center;
    font-size: 16px;
}

/* Centering and Sizing the Button */
.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    width: 200px; /* Specific width for centered look */
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    transform: scale(1.05);
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
# UI INPUT (CENTERED LAYOUT)
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>🎓 Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.8;'>Hybrid ML Model (Pass / Fail + Marks)</p>", unsafe_allow_html=True)

    # Input Fields
    sh = st.text_input("Study Hours", placeholder="Enter hours (e.g., 8)")
    at = st.text_input("Attendance %", placeholder="Enter % (e.g., 85)")
    
    # Large Centered Button
    predict = st.button("Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction Logic
# =============================
if predict:
    try:
        sh_val = float(sh)
        at_val = float(at)

        inp = scaler.transform([[sh_val, at_val]])
        pass_prob = log_model.predict_proba(inp)[0][1]
        marks = min(lin_model.predict(inp)[0], 100)

        color = "#22c55e" if pass_prob >= 0.5 else "#ef4444"
        result_text = "PASS" if pass_prob >= 0.5 else "FAIL"

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            components.html(f"""
            <div style="
                background:rgba(255,255,255,0.18);
                padding:30px;
                border-radius:25px;
                text-align:center;
                color:white;
                font-family: sans-serif;
                border: 2px solid {color};">
                <h2 style="margin:0;">Prediction Result</h2>
                <p style="font-size:18px;">Estimated Marks: <b>{marks:.1f}/100</b></p>
                <h1 style="color:{color}; font-size: 60px; margin:15px 0;">{result_text}</h1>
                <p>Confidence: {pass_prob*100:.1f}%</p>
            </div>
            """, height=280)
    except ValueError:
        st.error("Please enter valid numbers for both fields.")

# =============================
# Footer
# =============================
st.markdown(
    "<br><hr><center style='opacity:0.5;'>Predictor v2.6 | AI Analytics Dashboard</center>",
    unsafe_allow_html=True
)
