import shap
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.logger import logging
# Function to explain the model using SHAP values

logging.info("explain_model has started")
def explain_model(data_path, model_path):
    df = pd.read_csv(data_path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

    le = LabelEncoder()
    le.fit(df["diagnosis"])
    df["diagnosis"] = le.transform(df["diagnosis"])
    
    X = df.drop(columns=["diagnosis"])

    model, scaler = joblib.load(model_path)
    X_scaled = scaler.transform(X)
    
    logging.info("Calculating SHAP values...")

    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    print("Showing SHAP summary plot...")
    shap.summary_plot(shap_values, X, plot_type="bar")
