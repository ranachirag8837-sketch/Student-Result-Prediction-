import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components

# =============================
# 1. Page Configuration
# =============================
st.set_page_config(
    page_title="Student Result Prediction AI",
    layout="wide"
)

# =============================
# 2. Custom CSS
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
        background-color: rgba(255, 255, 255, 0.08);
        margin-top: 20px;
        text-align: center;
    }

    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.95);
        color: black;
        border-radius: 12px;
        height: 45px;
        text-align: center;
    }

    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 12px;
        padding: 10px;
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 3. Dataset & Model Training
# =============================
df = pd.DataFrame({
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [40,45,50,60,65,75,80,85,90,95],
    "ResultNumeric": [0,0,0,0,1,1,1,1,1,1],
    "TotalMarks": [25,30,38,45,55,68,75,82,88,95]
})

X = df[["StudyHours", "Attendance"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

log_model = LogisticRegression().fit(X_scaled, df["ResultNumeric"])
lin_model = LinearRegression().fit(X_scaled, df["TotalMarks"])

# =============================
# 4. Main UI
# =============================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.markdown('<div class="info-border-box">', unsafe_allow_html=True)

    st.markdown("""
    <h1>Student Result Prediction</h1>
    <p style="opacity:0.7;">Hybrid ML Model (Logistic + Linear Regression)</p>
    """, unsafe_allow_html=True)

    sh_input = st.text_input("Study Hours (per day)", "8")
    at_input = st.text_input("Attendance (%)", "85")

    predict = st.button("Predict Result")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 5. Prediction + Dynamic Output
# =============================
if predict:
    try:
        sh = float(sh_input)
        at = float(at_input)

        inp = scaler.transform([[sh, at]])

        pass_prob = log_model.predict_proba(inp)[0][1]
        marks = min(lin_model.predict(inp)[0], 100)

        status = "PASS" if pass_prob >= 0.5 else "FAIL"
        color = "#22c55e" if status == "PASS" else "#ef4444"

        # ---- Dynamic Logic ----
        level = (
            "Excellent" if pass_prob >= 0.8 else
            "Good" if pass_prob >= 0.6 else
            "Needs Improvement"
        )

        level_color = (
            "#22c55e" if pass_prob >= 0.8 else
            "#facc15" if pass_prob >= 0.6 else
            "#ef4444"
        )

        recommendation = (
            "You are doing great. Maintain consistency and revise weekly."
            if pass_prob >= 0.8 else
            "You are close to success. Focus more on weak subjects."
            if pass_prob >= 0.6 else
            "High risk detected. Increase study hours and attendance immediately."
        )

        progress = int(pass_prob * 100)

        with col2:
            # ---- Prediction Card ----
            components.html(f"""
            <div style="background:rgba(255,255,255,0.12);
                        padding:25px;
                        border-radius:20px;
                        text-align:center;">
                <h2>Prediction Result</h2>
                <p>Pass Probability: <b>{pass_prob*100:.1f}%</b></p>
                <p>Estimated Marks: <b>{marks:.1f}/100</b></p>
                <h1 style="color:{color};">{status}</h1>
            </div>
            """, height=260)

            # ---- IMAGE STYLE + DYNAMIC ADVANCED FEATURES ----
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #6a11cb, #2575fc);
                border-radius: 25px;
                padding: 35px;
                margin-top: 30px;
                color: white;
                box-shadow: 0 15px 30px rgba(0,0,0,0.35);
            ">
                <h2>Advanced Features & Recommendations</h2>

                <p><b>1. Personalized Study Plan:</b>
                <span style="color:{level_color}; font-weight:bold;"> {level}</span></p>

                <div style="background:rgba(255,255,255,0.25);
                            border-radius:10px; overflow:hidden; margin-bottom:15px;">
                    <div style="width:{progress}%;
                                background:{level_color};
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
                    border-left:5px solid {level_color};
                    border-radius:10px;">
                    <b>AI Recommendation:</b><br>
                    {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<h3 style='text-align:center;'>Study Hours vs Attendance</h3>",
                        unsafe_allow_html=True)

            fig, ax = plt.subplots()
            fig.patch.set_facecolor('#4B0082')
            ax.bar(df["StudyHours"], df["Attendance"], alpha=0.5)
            ax.bar(sh, at, width=0.4, label="Your Input")
            ax.set_xlabel("Study Hours")
            ax.set_ylabel("Attendance %")
            ax.legend()
            st.pyplot(fig)

    except ValueError:
        st.error("Please enter valid numeric values")

# =============================
# Footer
# =============================
st.markdown(
    "<center style='opacity:0.5;'>Predictor v2.3 | AI Analytics Dashboard</center>",
    unsafe_allow_html=True
)
