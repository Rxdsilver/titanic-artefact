from titanic_Rxdsilver.data import load_data, clean_data, prepare_data
from titanic_Rxdsilver.registry import save_model
from titanic_Rxdsilver.train import train_model, evaluate_model, optimize_model

from dotenv import load_dotenv
import mlflow
import os

def main():
    # Chargement des variables d'environnement
    load_dotenv()

    data_dir = os.environ.get("DATA_DIR")
    models_dir = os.environ.get("MODELS_DIR")

    print(f"DATA_DIR = {data_dir}")
    print(f"MODELS_DIR = {models_dir}")

    # Chargement et préparation des données
    df = load_data()
    df_cleaned = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_cleaned)

    # Définir l'expérience MLflow (nom visible dans l'interface)
    mlflow.set_experiment("Titanic Logistic Regression")

    # Lancer le modèle de base
    model, y_pred = train_model(X_train, y_train, X_test)
    evaluate_model(y_test, y_pred)
    
    mlflow.end_run()

    # Optimisation du modèle (loguée elle aussi)
    best_model = optimize_model(X_train, y_train)

    # Sauvegarde locale du modèle de base
    save_model(model, "logistic_model.pkl")
if __name__ == "__main__":
    main()
