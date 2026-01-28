import streamlit as st
import joblib
import pandas as pd
import base64

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Dark / Light Mode Toggle (Centered)
# -----------------------------
c1, c2, c3 = st.columns([3,2,3])
with c2:
    mode = st.toggle("🌙 Dark Mode")

# -----------------------------
# Load Success Sound (MP3/WAV)
# -----------------------------
def play_sound():
    audio_file = open("success.mp3", "rb")  # put success.mp3 in same folder
    audio_bytes = audio_file.read()
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# -----------------------------
# CSS Styling
# -----------------------------
st.markdown(f"""
<style>

.stApp {{
    background: linear-gradient(-45deg,
        {'#0f2027, #203a43, #2c5364' if mode else '#667eea, #764ba2, #43cea2'}
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Segoe UI', sans-serif;
}}

@keyframes gradientBG {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

.block-container {{
    background: {'rgba(0,0,0,0.45)' if mode else 'rgba(255,255,255,0.22)'};
    backdrop-filter: blur(16px);
    padding: 2rem;
    border-radius: 20px;
    max-width: 720px;
    margin: 40px auto;
    box-shadow: 0 10px 35px rgba(0,0,0,0.35);
}}

h1 {{
    text-align: center;
    color: white;
}}

.desc {{
    text-align: center;
    color: #f1f1f1;
}}

label {{
    color: white !important;
    font-weight: 600;
}}

div.stButton > button {{
    width: 100%;
    background: linear-gradient(to right, #43cea2, #185a9d);
    color: white;
    font-size: 17px;
    padding: 0.7em;
    border-radius: 14px;
    border: none;
}}

div.stButton > button:hover {{
    transform: scale(1.04);
    box-shadow: 0 0 18px rgba(67,206,162,0.8);
}}

/* 🎉 PASS Celebration */
.pass-card {{
    background: linear-gradient(135deg, #00f260, #0575e6);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: 700;
    color: white;
    animation: pop 0.6s ease-out;
    box-shadow: 0 0 30px rgba(0,255,200,0.8);
}}

@keyframes pop {{
    0% {{ transform: scale(0.7); opacity: 0; }}
    100% {{ transform: scale(1); opacity: 1; }}
}}

.prob-card {{
    background: rgba(255,255,255,0.25);
    padding: 14px;
    border-radius: 14px;
    margin-top: 14px;
    color: white;
    text-align: center;
}}

.reco {{
    background: rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 14px;
    margin-top: 15px;
    color: white;
}}

.footer {{
    text-align: center;
    color: #dddddd;
    font-size: 13px;
    margin-top: 25px;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Result Prediction System")
st.markdown("""
<div class="desc">
Pass / Fail Prediction<br>
with <b>Celebration + Recommendation System</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# -----------------------------
# Text Inputs
# -----------------------------
study_hours = st.text_input("📘 Study Hours per Day", placeholder="e.g. 5")
attendance = st.text_input("📊 Attendance Percentage", placeholder="e.g. 85")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Result"):
    try:
        study_hours = float(study_hours)
        attendance = float(attendance)

        if study_hours < 0 or attendance < 0 or attendance > 100:
            st.error("❌ Invalid input values")
        else:
            df = pd.DataFrame([[study_hours, attendance]],
                              columns=["StudyHours", "Attendance"])
            scaled = scaler.transform(df)
            pred = model.predict(scaled)
            prob = model.predict_proba(scaled)[0][1]

            st.divider()

            if pred[0] == 1:
                play_sound()  # 🔊 SUCCESS SOUND

                st.markdown(f"""
                <div class="pass-card">
                    🎉🎊 STUDENT WILL PASS 🎊🎉<br>
                    <span style="font-size:14px;font-weight:400;">
                    Excellent performance! Keep it up 🚀
                    </span>
                </div>

                <div class="prob-card">
                    📈 <b>Pass Probability:</b> {prob*100:.2f}%
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div class="reco">
                ✅ <b>Recommendations:</b>
                <ul>
                    <li>Maintain consistent study habits</li>
                    <li>Practice mock tests</li>
                    <li>Keep attendance above 80%</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.error("❌ STUDENT WILL FAIL")
                st.info(f"📉 Fail Probability: {(1-prob)*100:.2f}%")

    except:
        st.error("⚠️ Please enter numeric values only")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
Built with ❤️ using Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
