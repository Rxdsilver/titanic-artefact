import pytest
import pandas as pd
import numpy as np
from titanic_Rxdsilver.train import train_model, evaluate_model, optimize_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

@pytest.fixture
def dummy_data():
    X = pd.DataFrame({
        'feature1': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        'feature2': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    })
    y = pd.Series([0, 1, 1, 0, 1, 0, 1, 0, 1, 0]) 
    return X, y

def test_train_model(dummy_data):
    X, y = dummy_data
    model, y_pred = train_model(X, y, X)
    
    assert isinstance(model, LogisticRegression)
    assert len(y_pred) == len(X)

def test_evaluate_model_output(capsys, dummy_data):
    X, y = dummy_data
    _, y_pred = train_model(X, y, X)

    evaluate_model(y, y_pred)

    captured = capsys.readouterr()
    assert "Accuracy:" in captured.out
    assert "Classification Report:" in captured.out
    assert "Confusion Matrix:" in captured.out

def test_optimize_model(dummy_data):
    X, y = dummy_data
    grid = optimize_model(X, y)

    assert isinstance(grid, GridSearchCV)
    assert hasattr(grid, "best_params_")
    assert hasattr(grid, "best_score_")
