# do not run template.py because it will create new files again and all code will be lost 
# use template.py if you want to create files in new project

# files
import os

folders = [
    "breast-cancer-prediction/data/raw",
    "breast-cancer-prediction/data/processed",
    "breast-cancer-prediction/notebooks",
    "breast-cancer-prediction/src/preprocessing",
    "breast-cancer-prediction/src/models",
    "breast-cancer-prediction/src/utils",
    "breast-cancer-prediction/app",
    "breast-cancer-prediction/tests",
    "breast-cancer-prediction/config",
    "breast-cancer-prediction/Loggers"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create main files
files = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocessing.ipynb",
    "notebooks/03_modeling.ipynb",
    "notebooks/04_evaluation.ipynb",
    "src/preprocessing/clean_data.py",
    "src/models/train_model.py",
    "src/models/evaluate_model.py",
    "src/models/explain_model.py",
    "src/utils/helpers.py",
    "app/streamlit_app.py",
    "tests/test_model.py",
    "config/config.yaml",
    "requirements.txt",
    "run.py"
]

# Create files with placeholder content
for file in files:
    with open(f"breast-cancer-prediction/{file}", "w") as f:
        f.write(f"# {file} placeholder\n")