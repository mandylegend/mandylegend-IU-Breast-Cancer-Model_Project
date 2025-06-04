# Breast Cancer Prediction App - Streamlit
# Created by: [Mandar More]
# University Project: Predicting Breast Cancer using ML and Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


# Split data (train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Classifier

model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))

# Streamlit 

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

# Prediction
if st.button("🔮 Predict"):
    st.markdown("""
    <style>
    .main {
        background-color:purple;
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
importances = model.coef_[0]
top_indices = np.argsort(importances)[-10:]
top_features = [data.feature_names[i] for i in top_indices]
top_importances = importances[top_indices]
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances, color='skyblue')
plt.xlabel("Coefficient Value")
plt.title("Top 10 Feature Importance")
st.pyplot(plt)

# Footer
st.markdown("---")
st.markdown("Breast Cancer Prediction App - Streamlit")
st.markdown("Created by: Mandar More")
st.markdown("University Project: Predicting Breast Cancer using ML and Streamlit")

