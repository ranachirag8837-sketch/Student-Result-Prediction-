import streamlit as st
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Student Result Prediction (Hybrid)",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load / Upload Dataset
# -----------------------------
st.sidebar.header("📁 Dataset")

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv
")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.sidebar.success("CSV loaded from data folder")

else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload student_data.csv",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV uploaded successfully")

    else:
        st.sidebar.warning("CSV not found. Using sample data.")
        df = pd.DataFrame({
            "StudyHours": [1,2,3,4,5,6,7,8],
            "Attendance": [45,50,55,60,70,80,90,95],
            "ResultNumeric": [0,0,0,1,1,1,1,1]
        })

# -----------------------------
# REQUIRED COLUMNS CHECK
# -----------------------------
required_cols = ["StudyHours", "Attendance", "ResultNumeric"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ Missing required column: {col}")
        st.stop()

# -----------------------------
# HANDLE TotalMarks SAFELY
# -----------------------------
if "TotalMarks" not in df.columns:
    st.warning("⚠️ 'TotalMarks' column not found. Auto-generating marks.")

    df["TotalMarks"] = (
        df["StudyHours"] * 10 +
        df["Attendance"] * 0.5 +
        np.random.normal(0, 5, len(df))
    ).clip(0, 100)

# -----------------------------
# Features & Targets
# -----------------------------
X = df[["StudyHours", "Attendance"]]
y_class = df["ResultNumeric"]
y_score = df["TotalMarks"]

# -----------------------------
# Train Hybrid Model (LIVE)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hybrid_logistic = LogisticRegression()
hybrid_logistic.fit(X_scaled, y_class)

hybrid_linear = LinearRegression()
hybrid_linear.fit(X_scaled, y_score)

# -----------------------------
# UI
# -----------------------------
st.title("🎓 Student Result Prediction System (Hybrid Model)")

st.markdown("""
This system uses a **Hybrid Machine Learning Model**:

- **Logistic Regression** → Pass / Fail  
- **Linear Regression** → Marks Prediction  

The system automatically adapts to your dataset.
""")

st.divider()

# -----------------------------
# Inputs
# -----------------------------
study_hours = st.slider("📘 Study Hours (per day)", 0.0, 10.0, 4.0, 0.1)
attendance = st.slider("📊 Attendance (%)", 0.0, 100.0, 75.0, 1.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Result"):

    input_df = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["StudyHours", "Attendance"]
    )

    input_scaled = scaler.transform(input_df)

    pass_prob = hybrid_logistic.predict_proba(input_scaled)[0][1]
    predicted_marks = hybrid_linear.predict(input_scaled)[0]

    st.divider()

    if pass_prob >= 0.5 and predicted_marks >= 40:
        st.success("🎉 **PASS**")
    else:
        st.error("❌ **FAIL**")

    st.info(f"📈 Pass Probability: {pass_prob * 100:.2f}%")
    st.info(f"📝 Predicted Marks: {predicted_marks:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hybrid Machine Learning Model")

