import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned dataset
data = pd.read_csv("data/cleaned_data.csv")

# Features & target
X = data.drop("Outcome", axis=1)   # update column name if needed
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Save model
joblib.dump(model, "results/logistic_regression_model.pkl")

# Save evaluation report
with open("results/logistic_regression_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

print("Model and evaluation saved successfully!")
