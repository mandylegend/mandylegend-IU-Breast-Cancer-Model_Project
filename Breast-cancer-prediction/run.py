from src.models.train_model import train_and_save_model
from src.models.evaluate_model import evaluate_model
from src.models.explain_model import explain_model

 # Main script to run the training, evaluation, and explanation of the breast cancer prediction model
DATA_PATH = "E:/IU model engineering/Breast-cancer-prediction/data/raw/Cancer_data.csv"
MODEL_PATH = "src/models/logistic_model.pkl"

# this will plot both confusion matrix and shap values

if __name__ == "__main__":
    train_and_save_model(DATA_PATH,MODEL_PATH)
    evaluate_model(DATA_PATH, MODEL_PATH)
    explain_model(DATA_PATH, MODEL_PATH)
