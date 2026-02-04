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
    page_title="Student Result Prediction AI",
    layout="wide"
)

# =============================
# CSS
# =============================
st.markdown("""
<style>
.stApp {
    background-color: #4B0082;
    color: white;
}
.info-box {
    background: rgba(255,255,255,0.08);
    border-radius: 25px;
    padding: 35px;
    text-align: center;
}
input {
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
# UI
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("<h1>Student Result Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p>Hybrid ML Model</p>", unsafe_allow_html=True)

    sh = st.text_input("Study Hours", "8")
    at = st.text_input("Attendance %", "85")
    predict = st.button("Predict")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Prediction
# =============================
if predict:
    sh = float(sh)
    at = float(at)

    inp = scaler.transform([[sh, at]])
    pass_prob = log_model.predict_proba(inp)[0][1]
    marks = min(lin_model.predict(inp)[0], 100)

    # Dynamic logic
    if pass_prob >= 0.8:
        level = "Excellent"
        color = "#22c55e"
        advice = "You are doing great. Maintain consistency and revise weekly."
    elif pass_prob >= 0.6:
        level = "Good"
        color = "#facc15"
        advice = "You are close to success. Focus more on weak subjects."
    else:
        level = "Needs Improvement"
        color = "#ef4444"
        advice = "Increase study hours and attendance immediately."

    progress = int(pass_prob * 100)

    with col2:
        # Result Card
        components.html(f"""
        <div style="background:rgba(255,255,255,0.15);
                    padding:25px;
                    border-radius:20px;
                    text-align:center;">
            <h2>Prediction Result</h2>
            <p>Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
            <p>Estimated Marks: <b>{marks:.1f}/100</b></p>
            <h1 style="color:{color};">{'PASS' if pass_prob>=0.5 else 'FAIL'}</h1>
        </div>
        """, height=260)

        # ✅ FIXED ADVANCED FEATURES (NO HTML TEXT ERROR)
        components.html(f"""
        <div style="
            margin-top:30px;
            background:linear-gradient(135deg,#6a11cb,#2575fc);
            border-radius:25px;
            padding:35px;
            color:white;
            box-shadow:0 15px 30px rgba(0,0,0,0.35);
        ">
            <h2>Advanced Features & Recommendations</h2>

            <p><b>1. Personalized Study Plan:</b>
            <span style="color:{color}; font-weight:bold;"> {level}</span></p>

            <div style="background:rgba(255,255,255,0.25);
                        border-radius:10px;
                        overflow:hidden;
                        margin-bottom:15px;">
                <div style="width:{progress}%;
                            background:{color};
                            padding:6px;"></div>
            </div>

            <p><b>2. Topic-wise Analysis:</b> Detailed performance by subject.</p>
            <p><b>3. Peer Comparison:</b> Anonymous comparative statistics.</p>
            <p><b>4. Goal Setting & Tracking:</b> Smart progress monitoring.</p>
            <p><b>5. Tutor Connect:</b> AI or Human tutor support.</p>

            <div style="
                margin-top:20px;
                padding:15px;
                background:rgba(0,0,0,0.25);
                border-left:5px solid {color};
                border-radius:10px;">
                <b>AI Recommendation:</b><br>
                {advice}
            </div>
        </div>
        """, height=420)

        # Graph
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#4B0082')
        ax.bar(df["StudyHours"], df["Attendance"], alpha=0.5)
        ax.bar(sh, at, width=0.4)
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Attendance %")
        st.pyplot(fig)

st.markdown(
    "<center style='opacity:0.5;'>Predictor v2.3 | AI Analytics Dashboard</center>",
    unsafe_allow_html=True
)
