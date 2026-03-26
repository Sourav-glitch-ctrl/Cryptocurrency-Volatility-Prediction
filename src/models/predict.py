"""
predict.py
----------
Loads persisted model + scaler artifacts and produces predictions.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_artifacts(model_path: str, scaler_path: str = None, feature_names_path: str = None):
    """
    Load a trained model and its associated scaler from disk.

    Parameters
    ----------
    model_path  : Path to the saved model (.pkl).
    scaler_path : Path to the saved scaler (.pkl).
                  Defaults to 'scaler.pkl' in the same directory as model_path.
    feature_names_path : Path to the saved feature names list (.pkl).
                         Defaults to 'feature_names.pkl' in the same directory as model_path.

    Returns
    -------
    (model, scaler, feature_names)
    """
    model_path = Path(model_path)
    if scaler_path is None:
        scaler_path = str(model_path.parent / "scaler.pkl")

    if feature_names_path is None:
        feature_names_path = str(model_path.parent / "feature_names.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path) if Path(feature_names_path).exists() else None
    
    return model, scaler, feature_names


def predict(
    model,
    X: pd.DataFrame,
    scaler=None,
) -> np.ndarray:
    """
    Run predictions with an optional scaler transformation.

    Parameters
    ----------
    model  : Fitted sklearn-compatible estimator (e.g. XGBRegressor).
    X      : Feature DataFrame (columns must match training columns).
    scaler : Fitted StandardScaler (or None to skip scaling).

    Returns
    -------
    np.ndarray of predictions.
    """
    # Always convert to numpy to avoid sklearn feature-names warnings
    X_input = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    if scaler is not None:
        X_input = scaler.transform(X_input)

    return model.predict(X_input)