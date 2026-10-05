# Student Performance Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict a student's final academic score based on factors such as study hours, attendance, previous examination performance, and assignment completion.

A Linear Regression model is trained using student performance data and evaluated using standard regression metrics.

## Objectives

- Predict student final scores using Machine Learning.
- Analyze factors that influence academic performance.
- Visualize relationships between student attributes and final scores.
- Evaluate the performance of the trained model.
- Provide predictions for new student data.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Dataset

The dataset contains the following features:

| Feature | Description |
|---|---|
| study_hours | Average study hours per day |
| attendance | Student attendance percentage |
| previous_score | Previous examination score |
| assignments_completed | Number of completed assignments |
| final_score | Final examination score |

## Machine Learning Model

The project uses **Linear Regression** to predict the final score.

### Input Features

- Study Hours
- Attendance
- Previous Score
- Assignments Completed

### Output

- Predicted Final Score
- Performance Category

## Project Structure

```text
Student-Performance-Prediction/
│
├── data/
│   └── student_data.csv
│
├── notebooks/
│   └── student_performance.ipynb
│
├── src/
│   ├── student_performance.py
│   └── predict.py
│
├── student_performance_model.pkl
├── README.md
└── requirements.txt

How to Run
1. Install Dependencies
pip install -r requirements.txt

2. Train the Model
python src/student_performance.py

3. Make a Prediction
python src/predict.py

Enter the student's information when prompted.
Model Evaluation
The model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
Example
Example student input:
Study Hours: 7
Attendance: 85%
Previous Score: 70
Assignments Completed: 9

The program generates a predicted final score and assigns a performance category.
Future Improvements
- Use a larger real-world dataset.
- Compare multiple Machine Learning algorithms.
- Add a graphical user interface.
- Add more student-related features.
- Deploy the model as a web application.
Author
Student Performance Prediction — Machine Learning Project

Save it.

### Current status

Your ML project is now essentially **complete**:

```text
✅ Dataset
✅ Data analysis
✅ Visualizations
✅ ML model
✅ Model evaluation
✅ Saved trained model
✅ Prediction program
✅ Requirements
✅ README