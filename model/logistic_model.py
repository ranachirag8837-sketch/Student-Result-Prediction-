
import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------------------
# Project root directory (auto detect)
# -----------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# CSV full path
CSV_PATH = os.path.join(ROOT_DIR, "data", "student_data.csv")

# Check file exists
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")

# -----------------------------------------
# Load dataset
# -----------------------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------------------
# Features & target
# -----------------------------------------
X = df[["StudyHours", "Attendance"]]
y = df["ResultNumeric"]

# -----------------------------------------
# Scaling
# -----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------
# Train-test split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# Train model
# -----------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------
# Save model
# -----------------------------------------
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "logistic_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("✅ logistic_model.pkl created successfully")
