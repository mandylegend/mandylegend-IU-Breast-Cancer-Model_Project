import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Function to train a logistic regression model and save it along with the scaler
def train_and_save_model(data_path, model_path):
    df = pd.read_csv(data_path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    le = LabelEncoder()
    le.fit(df["diagnosis"])
    df["diagnosis"] = le.transform(df["diagnosis"])
    
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    predict = model.predict(X_test)  # To ensure the model is trained and can predict
    print([predict])  # To ensure the model has classes set
    accracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accracy:.4f}")
    print("Model training completed successfully.")
    print(f"Model classes: {model.classes_}")
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    print(f"Model number of features: {model.n_features_in_}")
    print(f"Model number of classes: {model.classes_}")
    print(f"Model number of iterations: {model.n_iter_}")
   

    



 
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, scaler), model_path)
    print(f"Model saved to {model_path}")


