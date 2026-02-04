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

/* Input Card */
.info-box {
    background: rgba(255,255,255,0.12);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
}

/* Text Input Box */
.stTextInput > div > div > input {
    background-color: white !important;
    color: black !important;
    border-radius: 12px;
    height: 45px;
    text-align: center;
    font-size: 16px;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
    border: none;
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
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='opacity:0.8;'>Hybrid ML Model (Pass/Fail + Marks Estimation)</p>", unsafe_allow_html=True)

    sh = st.text_input("Daily Study Hours", "8")
    at = st.text_input("Attendance Percentage (%)", "85")
    predict = st.button("Generate Prediction")

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

    # Dynamic styling
    if pass_prob >= 0.8:
        level, color = "Excellent", "#22c55e"
        advice = "Fantastic! Your preparation is solid. Keep revising weekly."
    elif pass_prob >= 0.6:
        level, color = "Good", "#facc15"
        advice = "You're on the right track. Focus on solving mock papers."
    else:
        level, color = "Needs Improvement", "#ef4444"
        advice = "Immediate attention required. Increase study hours and attendance."

    progress_width = int(pass_prob * 100)

    # =============================
    # Prediction Result Display
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
            font-family: 'Segoe UI', sans-serif;">
            <h2 style="margin:0;">Prediction Result</h2>
            <p style="font-size:18px; margin:10px 0;">Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
            <p style="font-size:18px; margin:0;">Estimated Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{color}; font-size:60px; margin:15px 0; font-weight:900;">
                {'PASS' if pass_prob >= 0.5 else 'FAIL'}
            </h1>
        </div>
        """, height=280)

    # =============================
    # Advanced Dashboard with % Bars
    # =============================
    components.html(f"""
    <div style="
        margin-top:20px;
        background:linear-gradient(135deg,#6a11cb,#2575fc);
        border-radius:30px;
        padding:45px;
        color:white;
        font-family: sans-serif;
        box-shadow:0 20px 40px rgba(0,0,0,0.3);
    ">
        <h1 style="margin-top:0;">Advanced Analytics & Recommendations</h1>
        <p style="font-size:18px;"><b>Status:</b> <span style="color:{color};">{level}</span></p>

        <div style="background:rgba(255,255,255,0.2); border-radius:15px; height:12px; margin-bottom:30px;">
            <div style="width:{progress_width}%; background:{color}; height:100%; border-radius:15px;"></div>
        </div>

        <h2 style="border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom:10px;">📌 Topic-wise Performance</h2>

        <div style="background:rgba(0,0,0,0.2); padding:20px; border-radius:15px; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; font-weight:bold;">
                <span>📘 Mathematics: <span style="font-weight:normal; opacity:0.8;">Concept clear, improve speed.</span></span>
                <span>75%</span>
            </div>
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:10px;">
                <div style="width:75%; background:#60a5fa; height:100%; border-radius:10px;"></div>
            </div>
        </div>

        <div style="background:rgba(0,0,0,0.2); padding:20px; border-radius:15px; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; font-weight:bold;">
                <span>💻 Programming: <span style="font-weight:normal; opacity:0.8;">Good logic, practice projects.</span></span>
                <span>88%</span>
            </div>
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:10px;">
                <div style="width:88%; background:#22c55e; height:100%; border-radius:10px;"></div>
            </div>
        </div>

        <div style="margin-top:30px; padding:20px; background:rgba(0,0,0,0.3); border-left:8px solid {color}; border-radius:15px;">
            <b style="font-size:20px;">AI Personalized Recommendation:</b><br><br>
            <span style="font-size:18px;">{advice}</span>
        </div>
    </div>
    """, height=620)

    # =============================
    # PERFORMANCE LINE CHART
    # =============================
    st.write("## 📈 Performance Growth Projection")
    
    # Generate Line Data
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#4B0082') # Match app background
    ax.set_facecolor('#4B0082')

    # Plot the general trend line
    ax.plot(df["StudyHours"], df["Marks"], color='#60a5fa', linewidth=3, marker='o', markerfacecolor='white', label='Average Growth Trend')
    
    # Plot the user's current position
    ax.scatter(sh_val, marks, color='#facc15', s=200, zorder=5, label='Your Current Position', edgecolor='white')
    ax.annotate(f"You: {marks:.1f} Marks", (sh_val, marks), xytext=(sh_val-1, marks+5), color='#facc15', fontweight='bold')

    # Styling the axes
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_xlabel("Study Hours", color='white', fontsize=12)
    ax.set_ylabel("Expected Marks", color='white', fontsize=12)
    ax.grid(color='white', alpha=0.1)
    ax.legend(facecolor='#4B0082', labelcolor='white')

    st.pyplot(fig)

# =============================
# Footer
# =============================
st.markdown(
    "<br><hr><center style='opacity:0.5; color:white;'>Predictor v2.6 | Data Science Student Portal</center>",
    unsafe_allow_html=True
)

