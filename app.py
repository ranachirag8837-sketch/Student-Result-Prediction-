import streamlit as st
import pandas as pd
import numpy as np
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
    sh = st.text_input("Daily Study Hours", "8")
    at = st.text_input("Attendance Percentage (%)", "85")
    predict = st.button("Generate Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

if predict:
    sh_val = float(sh)
    at_val = float(at)
    inp = scaler.transform([[sh_val, at_val]])
    pass_prob = log_model.predict_proba(inp)[0][1]
    marks = min(lin_model.predict(inp)[0], 100)

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

    # Result Display
    with col2:
        st.write("") 
        components.html(f"""
        <div style="background:rgba(255,255,255,0.18); padding:30px; border-radius:25px; text-align:center; color:white; font-family: sans-serif;">
            <h2 style="margin:0;">Prediction Result</h2>
            <p>Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
            <p>Estimated Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{color}; font-size:60px; margin:15px 0;">{'PASS' if pass_prob >= 0.5 else 'FAIL'}</h1>
        </div>
        """, height=280)

    # Advanced Dashboard with % Bars
    components.html(f"""
    <div style="margin-top:20px; background:linear-gradient(135deg,#6a11cb,#2575fc); border-radius:30px; padding:45px; color:white; font-family: sans-serif;">
        <h1>Advanced Analytics</h1>
        <p><b>Status:</b> <span style="color:{color};">{level}</span></p>
        <div style="background:rgba(255,255,255,0.2); border-radius:15px; height:12px; margin-bottom:30px;">
            <div style="width:{progress_width}%; background:{color}; height:100%; border-radius:15px;"></div>
        </div>
        <h2>📌 Topic Performance</h2>
        <div style="background:rgba(0,0,0,0.2); padding:20px; border-radius:15px; margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between;"><span>📘 Mathematics</span><span>75%</span></div>
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:10px;"><div style="width:75%; background:#60a5fa; height:100%; border-radius:10px;"></div></div>
        </div>
        <div style="background:rgba(0,0,0,0.2); padding:20px; border-radius:15px;">
            <div style="display:flex; justify-content:space-between;"><span>💻 Programming</span><span>88%</span></div>
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; height:8px; margin-top:10px;"><div style="width:88%; background:#22c55e; height:100%; border-radius:10px;"></div></div>
        </div>
    </div>
    """, height=500)

    # =============================
    # NEW PERFORMANCE CHART (Radar Chart)
    # =============================
    st.write("## 📊 Skill Analysis Chart")
    
    categories = ['Mathematics', 'Programming', 'Logic', 'Attendance', 'Study Consistency']
    # Sample data values for the user
    user_values = [75, 88, 80, at_val, (sh_val*10)] 
    target_values = [70, 70, 70, 75, 60]

    # Radar Chart Logic
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(user_values), endpoint=False)
    
    # Close the loop
    user_values = np.append(user_values, user_values[0])
    target_values = np.append(target_values, target_values[0])
    label_loc = np.append(label_loc, label_loc[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#4B0082')
    ax.set_facecolor('#ffffff10')

    # Plot User Data
    ax.plot(label_loc, user_values, label='Your Profile', color='#facc15', linewidth=2)
    ax.fill(label_loc, user_values, color='#facc15', alpha=0.3)

    # Plot Target Data
    ax.plot(label_loc, target_values, label='Average Student', color='white', linestyle='dashed', linewidth=1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(label_loc[:-1], categories, color='white', size=10)
    ax.tick_params(axis='y', colors='gray', labelsize=8)
    ax.grid(color='white', alpha=0.2)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Center the chart
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.pyplot(fig)

# Footer
st.markdown("<br><hr><center style='opacity:0.5; color:white;'>Predictor v2.6 | Data Science Student Portal</center>", unsafe_allow_html=True)
