import streamlit as st
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Student Result Prediction (Hybrid)",
    page_icon="🎓",
    layout="centered"
)

# =============================
# Sidebar – Dataset Loading
# =============================
st.sidebar.header("📁 Dataset")

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.sidebar.success("✅ CSV loaded from data folder")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload student_data.csv",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ CSV uploaded successfully")

    else:
        st.sidebar.warning("⚠️ CSV not found. Using sample data")
        df = pd.DataFrame({
            "StudyHours": [1,2,3,4,5,6,7,8],
            "Attendance": [45,50,55,60,70,80,90,95],
            "ResultNumeric": [0,0,0,1,1,1,1,1],
            "TotalMarks": [30,35,40,50,60,70,85,92]
        })

# =============================
# Required Columns Check
# =============================
required_columns = [
    "StudyHours",
    "Attendance",
    "ResultNumeric",
    "TotalMarks"
]

for col in required_columns:
    if col not in df.columns:
        st.error(f"❌ Missing required column: {col}")
        st.stop()

# =============================
# Feature & Target Selection
# =============================
X = df[["StudyHours", "Attendance"]]
y_class = df["ResultNumeric"]   # Pass / Fail
y_marks = df["TotalMarks"]      # Marks Prediction

# =============================
# Model Training (Hybrid)
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression → Pass/Fail
logistic_model = LogisticRegression()
logistic_model.fit(X_scaled, y_class)

# Linear Regression → Marks
linear_model = LinearRegression()
linear_model.fit(X_scaled, y_marks)

# =============================
# UI – Main Page
# =============================
st.title("🎓 Student Result Prediction System (Hybrid Model)")

st.markdown("""
### 🔹 Hybrid Machine Learning Model
- **Logistic Regression** → Pass / Fail  
- **Linear Regression** → Marks Prediction  

📌 The system automatically adapts to your dataset.
""")

st.divider()

# =============================
# User Inputs
# =============================
study_hours = st.slider(
    "📘 Study Hours (per day)",
    min_value=0.0,
    max_value=10.0,
    value=4.0,
    step=0.1
)

attendance = st.slider(
    "📊 Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

# =============================
# Prediction Button
# =============================
if st.button("🔍 Predict Result"):

    input_data = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["StudyHours", "Attendance"]
    )

    input_scaled = scaler.transform(input_data)

    pass_probability = logistic_model.predict_proba(input_scaled)[0][1]
    predicted_marks = linear_model.predict(input_scaled)[0]

    st.divider()

    # Final Decision
    if pass_probability >= 0.5 and predicted_marks >= 40:
        st.success("🎉 RESULT: **PASS**")
    else:
        st.error("❌ RESULT: **FAIL**")

    st.info(f"📈 Pass Probability: **{pass_probability * 100:.2f}%**")
    st.info(f"📝 Predicted Marks: **{predicted_marks:.2f} / 100**")

# =============================
# Footer
# =============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hybrid Machine Learning Model")
