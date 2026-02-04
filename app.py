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
# CSS (Centering Text and Large Button)
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}

/* Input Card Container */
.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 40px;
    text-align: center;
}

/* Centering Input Labels */
.stTextInput label {
    display: block !important;
    text-align: center !important;
    width: 100%;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600;
}

/* Centering Input Box Text */
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    height: 50px;
    text-align: center;
    font-size: 16px;
}

/* BIG Predict Button Styling */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    width: 100% !important;  /* Makes button wide */
    height: 60px !important; /* Makes button tall */
    font-size: 22px !important;
    font-weight: bold;
    border: none;
    margin-top: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# =============================
# Dataset & Models
# =============================
df = pd.DataFrame({
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 45, 50, 60, 65, 75, 80, 85, 90, 95],
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "Marks": [25, 30, 38, 45, 55, 68, 75, 82, 88, 95]
})

X = df[["StudyHours", "Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression().fit(X_scaled, df["Result"])
lin_model = LinearRegression().fit(X_scaled, df["Marks"])

# =============================
# UI Layout
# =============================
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-bottom:0;'>🎓 Student Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.8; margin-bottom:30px;'>Hybrid ML Model (Pass/Fail + Marks)</p>", unsafe_allow_html=True)

    # Input Fields (Labels are centered via CSS)
    sh = st.text_input("Study Hours", placeholder="Enter hours (e.g. 7)")
    at = st.text_input("Attendance %", placeholder="Enter percentage (e.g. 85)")
    
    # Large Predict Button
    predict = st.button("Predict Result")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction Output
# =============================
if predict:
    if sh and at:
        try:
            sh_val = float(sh)
            at_val = float(at)

            # Process Prediction
            inp = scaler.transform([[sh_val, at_val]])
            pass_prob = log_model.predict_proba(inp)[0][1]
            marks = min(lin_model.predict(inp)[0], 100)

            result_color = "#22c55e" if pass_prob >= 0.5 else "#ef4444"
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
                    font-family: Arial, sans-serif;
                    border: 2px solid {result_color};">
                    <h2 style="margin:0; opacity:0.9;">Prediction Summary</h2>
                    <p style="font-size:20px; margin:10px 0;">Estimated Marks: <b>{marks:.1f}/100</b></p>
                    <h1 style="color:{result_color}; font-size: 65px; margin:15px 0; letter-spacing:2px;">{result_text}</h1>
                    <p style="opacity:0.8;">Probability: {pass_prob*100:.1f}%</p>
                </div>
                """, height=300)
                
        except ValueError:
            st.error("Invalid Input! Please enter numbers only.")
    else:
        st.warning("Please fill in both fields before predicting.")

# =============================
# Footer
# =============================
st.markdown(
    "<br><hr><center style='opacity:0.5; font-size:14px;'>Predictor v2.6 | AI Analytics Dashboard</center>",
    unsafe_allow_html=True
)
