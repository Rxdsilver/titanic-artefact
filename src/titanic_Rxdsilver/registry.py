"""
Save and load models or preprocessors using pickle.
"""

import os
import pickle


def save_model(model: any, model_name: str):
    """
    Save a model to a file inside MODELS_DIR or current directory.

    Args:
        model: The model to save (any picklable object).
        model_name (str): The filename to use (e.g., 'model.pkl').
    """
    models_dir = os.environ.get("MODELS_DIR", ".")
    os.makedirs(models_dir, exist_ok=True)

    full_path = os.path.join(models_dir, model_name)
    with open(full_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {full_path}")


def load_model(model_name: str) -> any:
    """
    Load a model from a file in MODELS_DIR or current directory.

    Args:
        model_name (str): The filename of the saved model.

    Returns:
        The loaded model (any Python object).
    """
    models_dir = os.environ.get("MODELS_DIR", ".")
    full_path = os.path.join(models_dir, model_name)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")

    with open(full_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {full_path}")
    return model
