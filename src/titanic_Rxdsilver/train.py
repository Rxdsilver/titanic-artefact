"""
Train the Titanic model with MLflow tracking.
"""
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train, X_test):
    """
    Train a logistic regression model on the Titanic dataset.
    Logs parameters, metrics and the trained model using MLflow.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Labels for training.
        X_test (pd.DataFrame): Features for prediction.

    Returns:
        model: Trained LogisticRegression model.
        pd.Series: Predictions on X_test.
    """
    with mlflow.start_run(run_name="LogisticRegression-Baseline"):
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)

        # Log metrics
        acc = accuracy_score(y_train, model.predict(X_train))
        mlflow.log_metric("train_accuracy", acc)

        # Log model
        mlflow.sklearn.log_model(model, "logistic_model_baseline")

        return model, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the model's performance and log metrics with MLflow.

    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
    """
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("test_accuracy", acc)

    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def optimize_model(X_train, y_train):
    """
    Optimize the logistic regression model using GridSearchCV and log with MLflow.

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

    with mlflow.start_run(run_name="LogisticRegression-Optimized"):
        
        # Log best params
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("cv_folds", 5)

        # Log best CV accuracy
        mlflow.log_metric("best_cv_accuracy", grid.best_score_)

        # Log model
        mlflow.sklearn.log_model(grid.best_estimator_, "logistic_model_optimized")

    print("Best Parameters:", grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")

    return grid
