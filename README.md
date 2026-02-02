# 🎓 Student Result Prediction AI

A modern **Streamlit-based AI application** that predicts:
- ✅ Student **Pass / Fail probability**
- 📊 **Estimated marks (out of 100)**

The system uses a **Hybrid Machine Learning Model**
- Logistic Regression (Pass/Fail)
- Linear Regression (Marks Prediction)

---

## 🚀 Features

- Clean & modern UI (Custom CSS)
- Real-time prediction
- AI-based recommendations
- Interactive visualization
- Confetti animation on PASS 🎉

---

## 🧠 ML Models Used

- Logistic Regression
- Linear Regression
- StandardScaler (Feature Scaling)

---

## 🛠️ Tech Stack

- Python 3.9+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## 📂 Project Structure

Student_Result_Prediction_System/
│
├── data/
│   └── student_data.csv          # Student dataset (training data)
│
├── model/
│   ├── hybrid_linear.pkl         # Trained Linear Regression model
│   ├── hybrid_logistic.pkl       # Trained Logistic Regression model
│   ├── hybrid_scaler.pkl         # StandardScaler for hybrid model
│   ├── logistic_model.pkl        # Standalone logistic model
│   ├── logistic_model.py         # Model training script
│   └── scaler.pkl                # Feature scaler
│
├── utils/
│   └── app.py                    # Main Streamlit application
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
│
└── .gitignore (optional)         # Git ignored files


