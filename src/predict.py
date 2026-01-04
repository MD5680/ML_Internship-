import pandas as pd
import joblib

# Load trained model
model = joblib.load("results/logistic_regression_model.pkl")

# Example new patient data (change values as needed)
new_data = pd.DataFrame([{
    "Age": 45,
    "BP": 130,
    "Cholesterol": 200,
    "Glucose": 150
}])

# Prediction
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Disease Detected")
else:
    print("No Disease Detected")
