import os
import pytest
import pandas as pd
from titanic_Rxdsilver.data import load_data, clean_data, prepare_data

# --- Tests pour load_data ---

def test_load_data_success(monkeypatch, tmp_path):
    
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    
    # Crée un fichier CSV temporaire
    df = pd.DataFrame({
        "PassengerId": [1, 2],
        "Survived": [0, 1],
        "Pclass": [3, 1],
        "Name": ["Allen", "Smith"],
        "Sex": ["male", "female"],
        "Age": [22, 38],
        "SibSp": [1, 1],
        "Parch": [0, 0],
        "Ticket": ["A/5 21171", "PC 17599"],
        "Fare": [7.25, 71.2833],
        "Cabin": ["", "C85"],
        "Embarked": ["S", "C"]
    })
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=True)

    # Utilise le chemin temporaire
    loaded_df = load_data(data_dir=str(tmp_path))
    assert isinstance(loaded_df, pd.DataFrame)
    assert not loaded_df.empty
    assert "Survived" in loaded_df.columns

def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(data_dir=str(tmp_path))  # train.csv n'existe pas

def test_load_data_no_env_variable(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(ValueError):
        load_data()

# --- Tests pour clean_data ---

def test_clean_data_drops_and_encodes():
    df = pd.DataFrame({
        "Name": ["Allen", "Smith"],
        "Ticket": ["A/5 21171", "PC 17599"],
        "Cabin": ["", "C85"],
        "Sex": ["male", "female"],
        "Embarked": ["S", None],
        "Age": [22, None],
        "Survived": [0, 1]
    })
    cleaned_df = clean_data(df)
    assert "Name" not in cleaned_df.columns
    assert "Ticket" not in cleaned_df.columns
    assert "Cabin" not in cleaned_df.columns
    assert cleaned_df["Age"].isnull().sum() == 0
    assert cleaned_df["Embarked"].isnull().sum() == 0
    assert cleaned_df["Sex"].dtype in [int, "int32", "int64"]
    assert cleaned_df["Embarked"].dtype in [int, "int32", "int64"]

# --- Tests pour prepare_data ---

def test_prepare_data_splits_correctly():
    df = pd.DataFrame({
        "Survived": [0, 1, 0, 1],
        "Pclass": [3, 1, 2, 3],
        "Sex": [0, 1, 0, 1],
        "Age": [22, 38, 26, 35]
    })
    X_train, X_test, y_train, y_test = prepare_data(df)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert set(X_train.columns) == set(["Pclass", "Sex", "Age"])
