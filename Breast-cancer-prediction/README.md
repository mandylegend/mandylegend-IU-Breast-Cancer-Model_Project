🙏 Acknowledgments
UCI ML Repository - Breast Cancer Wisconsin Dataset

IU Internationale Hochschule – DLBDSME01 Model Engineering Module

# 🧠 Breast Cancer Prediction – Interpretable Machine Learning

This project builds an **interpretable machine learning model** to predict whether a breast tumor is **benign or malignant**, using the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

It follows the **CRISP-DM methodology** and includes a **Streamlit web app** for medical professionals to interact with the model and understand its predictions.

---

## 📌 Objective

- Develop a classification model with **F1 score > 0.95**
- Focus on **interpretability** using SHAP
- Allow **non-technical stakeholders** (e.g., oncologists) to understand model decisions
- Provide a prototype **GUI** for prediction and explanation

---

## 🧱 Project Structure

```bash
breast-cancer-prediction/
│
├── README.md                    # This file
├── LICENSE                      # Project license (optional)
├── .gitignore                   # Ignored files (e.g., data/, .pkl)
│
├── data/
│   ├── raw/                     # Original dataset (CSV)
│   └── processed/               # Scaled/cleaned data (optional)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   
│
├── src/
│   ├── preprocessing/
│   │   └── clean_data.py        # Cleaning functions
│   ├── models/
│   │   ├── train_model.py       # Train and save model
│   │   ├── evaluate_model.py    # Evaluate metrics and F1
│   │   └── explain_model.py     # SHAP-based explanations
│   └── utils/
│       └── helpers.py           # Utility functions
│
├── app/
│   └── Logistic_app.py         # GUI for predictions
    └── Ran_for_clf_app.py      # GUI for predictions
    
│
├── tests/
│   └── test_model.py            # Unit tests
│
├── config/
│   └── config.yaml              # Model parameters and paths
│
├── requirements.txt             # Python dependencies
└── run.py                       # Main runner (train + evaluate)

Important Notes

run.py - "it runs evalute_model.py , explain_model.py , train_model.py"
app - "it consists of two apps which are based on Logistic regression and RandomForestClassifier




