
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(filepath, target_column):
    # Load dataset
    data = pd.read_csv(C:\Users\MDAZAM\OneDrive\Desktop\internship)

    # Handle missing values
    data = data.dropna()

    # Encode categorical columns
    le = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col])

    # Split features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    return X, y
