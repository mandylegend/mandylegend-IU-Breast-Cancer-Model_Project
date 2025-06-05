# Breast Cancer Prediction App - Streamlit
# Created by: Mandar More
# University Project: Predicting Breast Cancer using ML and Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.logger import logging

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))


logging.info("Streamlit app has started")
# Streamlit UI
st.title("🩺 Breast Cancer Prediction App")
st.write("This app predicts whether a tumor is **Benign** or **Malignant** based on cell features.")

st.sidebar.header("🔍 Input Tumor Features")

# Let user input values for first 10 features
input_features = {}
for feature in data.feature_names[:10]:
    value = st.sidebar.number_input(f"{feature}", value=float(np.mean(X[feature])), step=0.1)
    input_features[feature] = value

# Fill the rest of the features with average values
for feature in data.feature_names[10:]:
    input_features[feature] = float(np.mean(X[feature]))

# Prepare input for prediction
input_df = pd.DataFrame([input_features])

st.markdown("""
    <style>
    .main {
        background-color:green;
    }
    .stButton>button {
        background-color: orange ;
        color: white;
    }
  

    h1 {
        color: pink;
    }
    </style>
""", unsafe_allow_html=True)

# Prediction
if st.button("🔮 Predict"):
    st.balloons()
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    st.subheader("🔬 Prediction Result")
    result = "🟢 Benign (Non-Cancerous)" if prediction == 1 else "🔴 Malignant (Cancerous)"
    st.success(f"Prediction: {result}")
    st.write(f"Prediction confidence: {round(np.max(prediction_proba) * 100, 2)}%")

# Show model accuracy
st.sidebar.subheader("📊 Model Info")
st.sidebar.write(f"Model Accuracy: **{round(accuracy * 100, 2)}%** on test data")

# Feature importance plot
st.subheader("📈 Feature Importance (Top 10)")
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:]
top_features = [data.feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

fig, ax = plt.subplots()
ax.barh(top_features, top_importances, color='skyblue')
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Project by **Mandar More** | Data Source: Breast Cancer Wisconsin Dataset (sklearn)")
