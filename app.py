import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
import streamlit.components.v1 as components  # Needed for JS confetti

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

# =============================
# Load Dataset
# =============================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    # Silent fallback dataset
    df = pd.DataFrame({
        "StudyHours": [1,2,3,4,5,6,7,8],
        "Attendance": [45,50,55,60,70,80,90,95],
        "ResultNumeric": [0,0,0,1,1,1,1,1],
        "TotalMarks": [30,35,40,50,60,70,85,92]
    })

# =============================
# Required Columns Check
# =============================
required_columns = ["StudyHours", "Attendance", "ResultNumeric", "TotalMarks"]
for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Missing required column: {col}")
        st.stop()

# =============================
# Feature & Target Selection
# =============================
X = df[["StudyHours", "Attendance"]]
y_class = df["ResultNumeric"]
y_marks = df["TotalMarks"]

# =============================
# Model Training (Hybrid)
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logistic_model = LogisticRegression()
logistic_model.fit(X_scaled, y_class)

linear_model = LinearRegression()
linear_model.fit(X_scaled, y_marks)

# =============================
# UI – Main Page
# =============================
st.title("🎓 Student Result Prediction System")
st.markdown("""
### 🔹 Hybrid Machine Learning Model
- **Logistic Regression** → Pass / Fail  
- **Linear Regression** → Marks Prediction  

📌 Clean UI • Internship ready
""")
st.divider()

# =============================
# User Inputs – Textboxes (allow empty)
# =============================
study_hours_input = st.text_input("📘 Study Hours (per day)", value="")
attendance_input = st.text_input("📊 Attendance (%)", value="")

# Convert inputs to float, handle empty input
try:
    study_hours = float(study_hours_input) if study_hours_input.strip() != "" else None
except:
    study_hours = None

try:
    attendance = float(attendance_input) if attendance_input.strip() != "" else None
except:
    attendance = None

# =============================
# Prediction + Recommendations + Confetti
# =============================
if st.button("🔍 Predict Result"):
    
    if study_hours is None or attendance is None:
        st.warning("⚠️ Please enter both Study Hours and Attendance before predicting.")
    else:
        input_data = pd.DataFrame([[study_hours, attendance]], columns=["StudyHours", "Attendance"])
        input_scaled = scaler.transform(input_data)

        pass_probability = logistic_model.predict_proba(input_scaled)[0][1]
        predicted_marks = linear_model.predict(input_scaled)[0]

        st.divider()

        # =============================
        # Result display + Confetti
        # =============================
        if pass_probability >= 0.5 and predicted_marks >= 40:
            st.success("🎉 RESULT: **PASS**")
            
            # 🎉 Confetti animation
            components.html("""
            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
            <script>
              setTimeout(() => {
                confetti({
                    particleCount: 300,
                    spread: 100,
                    origin: { y: 0.6 }
                });
              }, 100);
            </script>
            """, height=400)
        else:
            st.error("❌ RESULT: **FAIL**")

        st.info(f"📈 Pass Probability: **{pass_probability * 100:.2f}%**")
        st.info(f"📝 Predicted Marks: **{predicted_marks:.2f} / 100**")

        # =============================
        # Recommendations Logic
        # =============================
        st.markdown("### 💡 Recommendations:")
        if pass_probability < 0.5 or predicted_marks < 40:
            st.warning("""
            - Increase **study hours** per day.  
            - Improve **attendance** in classes.  
            - Focus on weak subjects or topics.  
            - Practice previous exams and exercises.
            """)
        elif predicted_marks < 60:
            st.info("""
            - Maintain or slightly increase study hours.  
            - Keep attendance high.  
            - Focus on revision and practice to improve marks.
            """)
        else:
            st.success("""
            - Excellent performance! 🎉  
            - Keep up the good work and continue consistent study habits.
            """)

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
