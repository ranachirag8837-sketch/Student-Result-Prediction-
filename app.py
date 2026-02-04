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
    st.markdown("<h1>Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.8;'>Hybrid ML Model (Pass/Fail + Marks Estimation)</p>", unsafe_allow_html=True)
    sh = st.text_input("Daily Study Hours", "9")
    at = st.text_input("Attendance Percentage (%)", "70")
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

    # DYNAMIC STATUS & COLOR LOGIC
    if pass_prob >= 0.8:
        status_text, status_color = "Excellent", "#22c55e"  # Green
        advice = "Fantastic! Your preparation is solid."
    elif pass_prob >= 0.5:
        status_text, status_color = "Good", "#facc15"       # Yellow
        advice = "You're on the right track."
    else:
        status_text, status_color = "Needs Improvement", "#ef4444" # Red
        advice = "Immediate attention required."

    result_big_text = "PASS" if pass_prob >= 0.5 else "FAIL"

    # =============================
    # 1. BIG TEXT RESULT (Dynamic)
    # =============================
    with col2:
        st.write("") 
        components.html(f"""
        <div style="
            background:rgba(255,255,255,0.15);
            padding:30px;
            border-radius:25px;
            text-align:center;
            color:white;
            font-family: 'Segoe UI', sans-serif;">
            <h2 style="margin:0;">Prediction Result</h2>
            <p style="font-size:18px; margin:10px 0;">Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
            <p style="font-size:18px; margin:0;">Estimated Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{status_color}; font-size:80px; margin:15px 0; font-weight:900; letter-spacing:2px;">
                {result_big_text}
            </h1>
        </div>
        """, height=320)

    # =============================
    # 2. STATUS & ANALYTICS (Dynamic)
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
        <h1 style="margin:0; font-size:28px;">Advanced Analytics & Recommendations</h1>
        <p style="font-size:20px; margin-top:10px;">
            <b>Status:</b> <span style="color:{status_color}; font-weight:bold;">{status_text}</span>
        </p>
        
        <div style="background:rgba(255,255,255,0.2); border-radius:15px; height:12px; margin-bottom:35px; margin-top:10px;">
            <div style="width:{int(pass_prob*100)}%; background:{status_color}; height:100%; border-radius:15px; box-shadow: 0 0 15px {status_color};"></div>
        </div>

        <h2 style="font-size:22px;">📌 Topic-wise Performance</h2>
        <hr style="opacity: 0.2; margin-bottom:20px;">
        
        <div style="margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between;">
                <span>📘 <b>Mathematics:</b> Concept clear, improve speed.</span>
                <span>{int(pass_prob*85)}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:8px;">
                <div style="width:{int(pass_prob*85)}%; background:#60a5fa; height:100%; border-radius:10px;"></div>
            </div>
        </div>

        <div style="margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between;">
                <span>💻 <b>Programming:</b> Good logic, practice projects.</span>
                <span>{int(pass_prob*90)}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:8px;">
                <div style="width:{int(pass_prob*90)}%; background:#22c55e; height:100%; border-radius:10px;"></div>
            </div>
        </div>

        <div style="margin-top:30px; padding:20px; background:rgba(0,0,0,0.2); border-left:8px solid {status_color}; border-radius:15px;">
            <b style="font-size:18px;">AI Recommendation:</b><br>
            <span>{advice}</span>
        </div>
    </div>
    """, height=580)

    # =============================
    # 3. PERFORMANCE LINE CHART
    # =============================
    st.write("## 📈 Marks Trend vs Study Hours")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('Balck') # Match app background
    ax.set_facecolor('#4B0082')

    # Plot trend line from data
    ax.plot(df["StudyHours"], df["Marks"], color='#60a5fa', linewidth=3, marker='o', markerfacecolor='white', label='Average Growth')
    
    # Highlight the current prediction point
    ax.scatter(sh_val, marks, color=status_color, s=200, zorder=5, label='Your Prediction', edgecolor='white')
    ax.annotate(f"{marks:.1f}", (sh_val, marks), xytext=(sh_val, marks+5), color='white', fontweight='bold', ha='center')

    # Styling axes
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel("Study Hours", color='white')
    ax.set_ylabel("Expected Marks", color='white')
    ax.grid(color='white', alpha=0.1)
    ax.legend(facecolor='#4B0082', labelcolor='white')

    st.pyplot(fig)

st.markdown("<br><hr><center style='opacity:0.3; color:white;'>Predictor AI v2.6</center>", unsafe_allow_html=True)

