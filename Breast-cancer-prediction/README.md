**do not run template.py because it will create new files and folder again and all code will be lost use template.py if you want to create files in another project**


🙏 Acknowledgments
UCI ML Repository - Breast Cancer Wisconsin Dataset

IU Internationale Hochschule – DLBDSME01 Model Engineering Module

# 🧠 Breast Cancer Prediction – Interpretable Machine Learning

This project builds an **interpretable machine learning model** to predict whether a breast tumor is **benign or malignant**, using the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) and IU csv file.

It follows the **CRISP-DM methodology** and includes a **Streamlit web app** for medical professionals to interact with the model and understand its predictions.

---

## 📌 Objective

- Develop a classification model with **F1 score > 0.95**
- Focus on **interpretability** using SHAP
- Allow **non-technical stakeholders** (e.g., oncologists) to understand model decisions
- Provide a prototype **GUI** for prediction and explanation



# libraries (you can see in requirement.txt)
pandas
numpy
scikit-learn
joblib
streamlit
shap
matplotlib
seaborn
pyyaml



# Important Notes

run.py - **it runs evaluate_model.py , explain_model.py , train_model.py**
app - **it consists of two apps which are based on Logistic regression and RandomForestClassifier**
notebooks - **this notebook consist of eda.ipynb and model.ipynb**
data - **csv file for training model**
Logger - **I have created logger file to keep track of code which is logs folder**



# How to run 
You need to install necessary  **libraries** which i have written in requirement.txt to run the project or you can create a new **enviroment**
To see result in a form of browser you need to go in **data/reports/breast_cancer_profiling_report.html** and from there you need to run live server
       

    

**github project link**
https://github.com/mandylegend/mandylegend-IU-Breast-Cancer-Model_Project/tree/main - you can fetch project from here and updated files

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
│   └── reports/               # HTML file
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


