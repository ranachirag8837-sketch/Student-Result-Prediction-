# ==========================================
# Advanced Student Result Prediction Web App
# Using Streamlit + Machine Learning
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Student Result Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Result Prediction System")
st.write("Advanced Machine Learning Web Application")

# ==========================================
# STEP 1: DATA COLLECTION
# ==========================================
data = pd.read_csv("student_data.csv")

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================
data = data.dropna()

# ==========================================
# STEP 3: FEATURE SCALING
# ==========================================
X = data.drop("FinalResult", axis=1)
y = data["FinalResult"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================================
# STEP 5: MODEL TRAINING (MULTIPLE MODELS)
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

accuracy = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracy[name] = acc

# ==========================================
# STEP 6: BEST MODEL SELECTION
# ==========================================
best_model_name = max(accuracy, key=accuracy.get)
best_model = models[best_model_name]

# ==========================================
# SIDEBAR - MODEL INFO
# ==========================================
st.sidebar.header("📊 Model Performance")
for model_name, acc in accuracy.items():
    st.sidebar.write(f"{model_name}: {acc:.2f}")

st.sidebar.success(f"Best Model: {best_model_name}")

# ==========================================
# USER INPUT SECTION
# ==========================================
st.subheader("📝 Enter Student Details")

study_hours = st.number_input("Study Hours per Day", 0.0, 12.0, 5.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0)
internal_marks = st.number_input("Internal Marks", 0.0, 100.0, 60.0)
assignment_score = st.number_input("Assignment Score", 0.0, 100.0, 70.0)
prev_result = st.selectbox("Previous Result", ["Fail", "Pass"])

prev_result_value = 1 if prev_result == "Pass" else 0

# ==========================================
# PREDICTION
# ==========================================
if st.button("🔍 Predict Result"):
    input_data = np.array([[study_hours, attendance, internal_marks,
                             assignment_score, prev_result_value]])

    input_scaled = scaler.transform(input_data)

    prediction = best_model.predict(input_scaled)[0]
    probability = best_model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Student will PASS (Probability: {probability*100:.2f}%)")
    else:
        st.error(f"❌ Student will FAIL (Probability: {(1-probability)*100:.2f}%)")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Developed using Streamlit & Machine Learning")
