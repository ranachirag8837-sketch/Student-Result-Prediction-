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
# Custom CSS
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
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Dataset & Training
# =============================
df = pd.DataFrame({
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [40,45,50,60,65,75,80,85,90,95],
    "ResultNumeric": [0,0,0,0,1,1,1,1,1,1],
    "TotalMarks": [25,30,38,45,55,68,75,82,88,95]
})

X = df[["StudyHours","Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression().fit(X_scaled, df["ResultNumeric"])
lin_model = LinearRegression().fit(X_scaled, df["TotalMarks"])

# =============================
# Layout
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-border-box">', unsafe_allow_html=True)
    st.markdown("<h1>🎓 Student Result Prediction</h1>", unsafe_allow_html=True)

    sh = st.text_input("📘 Study Hours", "8")
    at = st.text_input("📊 Attendance (%)", "85")
    predict = st.button("🌟 Predict Result")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction
# =============================
if predict:
    sh = float(sh)
    at = float(at)

    inp = pd.DataFrame([[sh,at]], columns=["StudyHours","Attendance"])
    inp_scaled = scaler.transform(inp)

    pass_prob = log_model.predict_proba(inp_scaled)[0][1]
    marks = min(float(lin_model.predict(inp_scaled)[0]),100)

    st.session_state["pass_prob"] = pass_prob
    st.session_state["marks"] = marks

    result = "PASS" if pass_prob>=0.5 else "FAIL"

    st.success(f"Result: {result}")
    st.info(f"Pass Probability: {pass_prob*100:.1f}%")
    st.info(f"Estimated Marks: {marks:.1f}/100")

# =============================
# 🤖 AI CHAT SECTION
# =============================
st.markdown("## 💬 AI Student Chat Assistant")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_msg = st.chat_input("Ask anything about study, marks, result...")

if user_msg:
    st.session_state.chat.append({"role":"user","content":user_msg})

    # AI Logic
    if "improve" in user_msg.lower():
        reply = "📈 Increase study hours by 1–2 hours and maintain attendance above 85%."
    elif "pass" in user_msg.lower():
        reply = "✅ Focus on consistency. Regular revision is key to passing."
    elif "fail" in user_msg.lower():
        reply = "⚠️ You need to improve both attendance and daily study routine."
    elif "marks" in user_msg.lower():
        reply = f"📊 Your estimated marks are {st.session_state.get('marks','N/A')}."
    else:
        reply = "🤖 I can help you with study tips, marks, attendance & improvement plans."

    st.session_state.chat.append({"role":"assistant","content":reply})

    with st.chat_message("assistant"):
        st.markdown(reply)

# Footer
st.markdown("<center style='opacity:0.6;'>Predictor v3.0 | AI + Chat Enabled</center>", unsafe_allow_html=True)
