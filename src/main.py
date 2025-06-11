# src/main.py (ou src/titanic/main.py selon ton arborescence)

from titanic_Rxdsilver.data import load_data, clean_data, prepare_data
from titanic_Rxdsilver.registry import save_model
from titanic_Rxdsilver.train import train_model, evaluate_model, optimize_model

from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    data_dir = os.environ.get("DATA_DIR")
    models_dir = os.environ.get("MODELS_DIR")

    print(f"DATA_DIR = {data_dir}")
    print(f"MODELS_DIR = {models_dir}")

    df = load_data()
    df_cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_cleaned)

    model, y_pred = train_model(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred)

    best_model = optimize_model(X_train, y_train)
    save_model(model, "logistic_model.pkl")

if __name__ == "__main__":
    main()
