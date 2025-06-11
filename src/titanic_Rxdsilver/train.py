"""
Train the Titanic model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train, X_test):
    """
    Train a logistic regression model on the Titanic dataset.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Labels for training.
        X_test (pd.DataFrame): Features for prediction.

    Returns:
        model: Trained LogisticRegression model.
        pd.Series: Predictions on X_test.
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the model's performance.

    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
    """
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def optimize_model(X_train, y_train):
    """
    Optimize the logistic regression model using GridSearchCV.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        GridSearchCV: The fitted GridSearchCV object with best parameters.
    """
    param_grid = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}

    grid = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    return grid
