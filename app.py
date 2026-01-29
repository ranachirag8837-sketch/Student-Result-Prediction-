import streamlit as st
import os
import joblib
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
# Paths
# -----------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

LOGISTIC_PATH = os.path.join(MODEL_DIR, "hybrid_logistic.pkl")
LINEAR_PATH = os.path.join(MODEL_DIR, "hybrid_linear.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "hybrid_scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Auto train model if not exists
# -----------------------------
if not (os.path.exists(LOGISTIC_PATH) and os.path.exists(LINEAR_PATH) and os.path.exists(SCALER_PATH)):

    df = pd.read_csv(DATA_PATH)

    X = df[["StudyHours", "Attendance"]]
    y_class = df["ResultNumeric"]   # 0 = Fail, 1 = Pass
    y_score = df["TotalMarks"]      # Marks

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hybrid_logistic = LogisticRegression()
    hybrid_logistic.fit(X_scaled, y_class)

    hybrid_linear = LinearRegression()
    hybrid_linear.fit(X_scaled, y_score)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(hybrid_logistic, LOGISTIC_PATH)
    joblib.dump(hybrid_linear, LINEAR_PATH)

# -----------------------------
# Load models
# -----------------------------
scaler = joblib.load(SCALER_PATH)
hybrid_logistic = joblib.load(LOGISTIC_PATH)
hybrid_linear = joblib.load(LINEAR_PATH)

# -----------------------------
# UI
# -----------------------------
st.title("🎓 Student Result Prediction System (Hybrid Model)")

st.markdown("""
This system uses a **Hybrid Machine Learning Model**:

- **Logistic Regression** → Pass / Fail Probability  
- **Linear Regression** → Predicted Marks  

Final result is based on **both models together**.
""")

st.divider()

# -----------------------------
# Inputs
# -----------------------------
study_hours = st.slider(
    "📘 Study Hours (per day)",
    0.0, 10.0, 4.0, 0.1
)

attendance = st.slider(
    "📊 Attendance (%)",
    0.0, 100.0, 75.0, 1.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Result"):

    input_data = pd.DataFrame(
        [[study_hours, attendance]],
        columns=["StudyHours", "Attendance"]
    )

    input_scaled = scaler.transform(input_data)

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
