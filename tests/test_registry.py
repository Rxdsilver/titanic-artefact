import os
import tempfile
import pandas as pd
from titanic_Rxdsilver.data import load_data, clean_data, prepare_data
from titanic_Rxdsilver.train import train_model
from titanic_Rxdsilver.registry import save_model, load_model


def test_save_and_load_model():
    with tempfile.TemporaryDirectory() as tmp_data_dir, tempfile.TemporaryDirectory() as tmp_models_dir:
        # Création d'un train.csv minimal dans tmp_data_dir
        df = pd.DataFrame({
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 0, 1],
            "Pclass": [3, 1, 3, 2],
            "Name": ["Allen", "Smith", "Brown", "Davis"],
            "Sex": ["male", "female", "male", "female"],
            "Age": [22, 38, 26, 35],
            "SibSp": [1, 1, 0, 0],
            "Parch": [0, 0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "347082", "345678"],
            "Fare": [7.25, 71.2833, 8.05, 12.35],
            "Cabin": ["", "C85", "", ""],
            "Embarked": ["S", "C", "S", "S"]
        })
        csv_path = os.path.join(tmp_data_dir, "train.csv")
        df.to_csv(csv_path, index=True)

        # Patch les variables d'environnement nécessaires
        os.environ["DATA_DIR"] = tmp_data_dir
        os.environ["MODELS_DIR"] = tmp_models_dir

        # Charge, nettoie et prépare les données
        data = clean_data(load_data())
        X_train, X_test, y_train, y_test = prepare_data(data)

        # Entraîne le modèle
        model, _ = train_model(X_train, y_train, X_test)

        # Sauvegarde et recharge le modèle
        save_model(model, "test_model.pkl")
        loaded_model = load_model("test_model.pkl")

        assert hasattr(loaded_model, "predict")
