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
# CSS (Custom Styling for Centering and Big Button)
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}

/* Container for the Input Box */
.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
}

/* Centering the Labels of Text Input */
.stTextInput label {
    display: block;
    text-align: center;
    width: 100%;
    color: white !important;
    font-size: 18px !important;
    font-weight: bold;
}

/* Styling the Input Box itself */
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    height: 50px;
    text-align: center;
    font-size: 16px;
}

/* Making the Predict Button BIG (Full Width) */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    width: 100%;  /* બટનને મોટું કરવા માટે */
    height: 55px; /* બટનની ઉંચાઈ વધારવા માટે */
    font-size: 20px;
    font-weight: bold;
    border: none;
    margin-top: 20px;
    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    transform: scale(1.02);
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
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🎓 Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8;'>Hybrid ML Model (Pass / Fail + Marks)</p>", unsafe_allow_html=True)

    # Inputs
    sh = st.text_input("Study Hours", placeholder="e.g. 5")
    at = st.text_input("Attendance %", placeholder="e.g. 75")
    
    # Big Button
    predict = st.button("Predict Now")
    
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction Result Section
# =============================
if predict:
    if sh and at:
        try:
            sh_val = float(sh)
            at_val = float(at)

            inp = scaler.transform([[sh_val, at_val]])
            pass_prob = log_model.predict_proba(inp)[0][1]
            marks = min(lin_model.predict(inp)[0], 100)

            color = "#22c55e" if pass_prob >= 0.5 else "#ef4444"
            status = "PASS" if pass_prob >= 0.5 else "FAIL"

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
                    <h2 style="margin:0;">Result Overview</h2>
                    <p style="font-size:18px;">Marks Estimate: <b>{marks:.1f}/100</b></p>
                    <h1 style="color:{color}; font-size: 60px; margin:10px 0;">{status}</h1>
                    <p>Confidence: {pass_prob*100:.1f}%</p>
                </div>
                """, height=280)
        except ValueError:
            st.error("કૃપા કરીને ફક્ત નંબર દાખલ કરો.")
    else:
        st.warning("બંને ખાનામાં વિગતો ભરો.")

# =============================
# Footer
# =============================
st.markdown(
    "<br><hr><center style='opacity:0.5;'>Predictor v2.6 | Powered by AI Analytics</center>",
    unsafe_allow_html=True
)
