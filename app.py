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
# CSS (Custom Styling)
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}
.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
}
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    height: 45px;
    text-align: center;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 45px;
    width: 100%;
    font-weight: bold;
    border: none;
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
# UI INPUT
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>Student Result Prediction</h1>")
    sh = st.text_input("Daily Study Hours", "8")
    at = st.text_input("Attendance Percentage (%)", "85")
    predict = st.button("Generate Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Dynamic Prediction Logic
# =============================
if predict:
    sh_val = float(sh)
    at_val = float(at)

    inp = scaler.transform([[sh_val, at_val]])
    pass_prob = log_model.predict_proba(inp)[0][1]
    marks = min(lin_model.predict(inp)[0], 100)

    # Logic for Dynamic Colors and Status
    is_pass = pass_prob >= 0.5
    result_text = "PASS" if is_pass else "FAIL"
    main_color = "#22c55e" if is_pass else "#ef4444" # Green if Pass, Red if Fail
    
    # Topic-specific colors (Dynamic based on Pass/Fail)
    math_color = "#60a5fa" if is_pass else "#f87171"
    prog_color = "#22c55e" if is_pass else "#f87171"
    
    # Topic-specific feedback text
    math_feedback = "Concept clear, improve speed." if is_pass else "Weak concepts, need basics."
    prog_feedback = "Good logic, practice projects." if is_pass else "Logic errors, need more coding."

    # =============================
    # Section 1: Prediction Result (Dynamic)
    # =============================
    with col2:
        st.write("") 
        components.html(f"""
        <div style="
            background:rgba(255,255,255,0.18);
            padding:30px;
            border-radius:25px;
            text-align:center;
            color:white;
            font-family: sans-serif;">
            <h2 style="margin:0;">Prediction Result</h2>
            <p style="font-size:18px; margin:10px 0;">Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
            <p style="font-size:18px; margin:0;">Estimated Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{main_color}; font-size:70px; margin:15px 0; font-weight:900;">
                {result_text}
            </h1>
        </div>
        """, height=280)

    # =============================
    # Section 2: Topic-wise Performance (Dynamic)
    # =============================
    components.html(f"""
    <div style="
        margin-top:20px;
        background:linear-gradient(135deg,#6a11cb,#2575fc);
        border-radius:30px;
        padding:40px;
        color:white;
        font-family: sans-serif;
    ">
        <h2 style="margin-top:0;">📌 Topic-wise Performance</h2>
        <hr style="opacity: 0.3; margin-bottom: 25px;">
        
        <div style="margin-bottom:25px;">
            <div style="display:flex; justify-content:space-between; align-items: center;">
                <span>📘 <b>Mathematics:</b> {math_feedback}</span>
                <span style="font-weight:bold;">{int(pass_prob * 85)}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:10px; height:10px; margin-top:10px;">
                <div style="width:{int(pass_prob * 85)}%; background:{math_color}; height:100%; border-radius:10px; box-shadow: 0 0 10px {math_color};"></div>
            </div>
        </div>

        <div style="margin-bottom:25px;">
            <div style="display:flex; justify-content:space-between; align-items: center;">
                <span>💻 <b>Programming:</b> {prog_feedback}</span>
                <span style="font-weight:bold;">{int(pass_prob * 90)}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.15); border-radius:10px; height:10px; margin-top:10px;">
                <div style="width:{int(pass_prob * 90)}%; background:{prog_color}; height:100%; border-radius:10px; box-shadow: 0 0 10px {prog_color};"></div>
            </div>
        </div>
    </div>
    """, height=400)

    # =============================
    # Line Chart
    # =============================
    st.write("## 📈 Performance Growth Projection")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#000000') 
    ax.set_facecolor('#4B0082')
    ax.plot(df["StudyHours"], df["Marks"], color='#60a5fa', linewidth=3, marker='o', markerfacecolor='white')
    ax.scatter(sh_val, marks, color=main_color, s=200, zorder=5, edgecolor='white')
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('white')
    st.pyplot(fig)

st.markdown("<br><hr><center style='opacity:0.5; color:white;'>Predictor v2.6 | AI Student Dashboard</center>", unsafe_allow_html=True)
