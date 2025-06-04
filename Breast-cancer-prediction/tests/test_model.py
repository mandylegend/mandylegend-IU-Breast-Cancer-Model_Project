from src.models.train_model import train_and_save_model
import os

# Test for the training pipeline of the breast cancer prediction model
def test_training_pipeline():
    model_path = "E:\IU model engineering\src\models\logistic_model.pkl"
    train_and_save_model("data/raw/Cancer_data.csv", model_path)
    assert os.path.exists(model_path), "Model was not saved!"
