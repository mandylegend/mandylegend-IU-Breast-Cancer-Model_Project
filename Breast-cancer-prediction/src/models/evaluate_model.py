from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from src.logger import logging

logging.info("evaluate has started")


# Function to evaluate the model using a dataset and a saved model
def evaluate_model(data_path, model_path):
    df = pd.read_csv(data_path)
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

    df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    _, scaler = joblib.load(model_path)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    model, _ = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print("Evaluation Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = range(len(model.classes_))
    plt.xticks(tick_marks, model.classes_, rotation=45)
    plt.yticks(tick_marks, model.classes_)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    # plt.savefig("confusion_matrix.png")
    plt.show()
    # print("Confusion matrix saved as 'confusion_matrix.png'.")

    



