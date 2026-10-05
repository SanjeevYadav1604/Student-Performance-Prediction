# ============================================
# STUDENT PERFORMANCE PREDICTION
# Machine Learning Project
# ============================================

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================
# 1. LOAD DATASET
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

# ============================================
# 2. CHECK DATA
# ============================================

print("\nDataset information:")
print(data.info())

print("\nMissing values:")
print(data.isnull().sum())

print("\nStatistical summary:")
print(data.describe())

# ============================================
# 3. DATA VISUALIZATION
# ============================================

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="study_hours",
    y="final_score"
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data,
    x="attendance",
    y="final_score"
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation")
plt.show()

# ============================================
# 4. PREPARE DATA
# ============================================

X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments_completed"
    ]
]

y = data["final_score"]

# ============================================
# 5. SPLIT DATASET
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================
# 6. TRAIN MACHINE LEARNING MODEL
# ============================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")
# Save trained model
MODEL_PATH = os.path.join(BASE_DIR, "student_performance_model.pkl")
joblib.dump(model, MODEL_PATH)

print(f"Model saved successfully: {MODEL_PATH}")

# ============================================
# 7. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": np.round(y_pred, 2)
})

print("\nPrediction Results:")
print(results)

# ============================================
# 8. MODEL EVALUATION
# ============================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("----------------------------")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"Root Mean Square Error : {rmse:.2f}")
print(f"R² Score            : {r2:.2f}")

# ============================================
# 9. ACTUAL VS PREDICTED GRAPH
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.title("Actual vs Predicted Final Scores")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")

plt.show()

# ============================================
# 10. PREDICT PERFORMANCE OF A NEW STUDENT
# ============================================

new_student = pd.DataFrame({
    "study_hours": [7],
    "attendance": [85],
    "previous_score": [70],
    "assignments_completed": [9]
})

prediction = model.predict(new_student)[0]

print("\nNEW STUDENT PREDICTION")
print("----------------------------")
print(f"Study Hours: 7")
print(f"Attendance: 85%")
print(f"Previous Score: 70")
print(f"Assignments Completed: 9")
print(f"\nPredicted Final Score: {prediction:.2f}")

# ============================================
# 11. PERFORMANCE CATEGORY
# ============================================

if prediction >= 85:
    category = "Excellent"
elif prediction >= 70:
    category = "Good"
elif prediction >= 50:
    category = "Average"
else:
    category = "Needs Improvement"

print(f"Performance Category: {category}")

print("\n============================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("============================================")