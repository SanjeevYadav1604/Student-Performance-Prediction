import os
import joblib
import pandas as pd

# Find project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load trained model
MODEL_PATH = os.path.join(BASE_DIR, "student_performance_model.pkl")
model = joblib.load(MODEL_PATH)

print("\n======================================")
print("   STUDENT PERFORMANCE PREDICTOR")
print("======================================\n")

# Get student information
study_hours = float(input("Enter study hours per day: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous exam score: "))
assignments_completed = int(input("Enter assignments completed (out of 10): "))

# Create input data
student = pd.DataFrame({
    "study_hours": [study_hours],
    "attendance": [attendance],
    "previous_score": [previous_score],
    "assignments_completed": [assignments_completed]
})

# Make prediction
prediction = model.predict(student)[0]

# Keep score within 0-100
prediction = max(0, min(100, prediction))

# Determine category
if prediction >= 85:
    category = "Excellent"
elif prediction >= 70:
    category = "Good"
elif prediction >= 50:
    category = "Average"
else:
    category = "Needs Improvement"

# Display result
print("\n======================================")
print("           PREDICTION RESULT")
print("======================================")

print(f"Predicted Final Score : {prediction:.2f}/100")
print(f"Performance Category  : {category}")

print("======================================")