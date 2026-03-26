"""
train.py
--------
Trains an XGBoostRegressor on the feature matrix, fits a StandardScaler,
and persists both artifacts to disk using joblib.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_save_path: str,
    scaler_save_path: str = None,
    feature_names_save_path: str = None,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> tuple:
    """
    Fit a StandardScaler + XGBRegressor and save both to disk.

    Parameters
    ----------
    X_train          : Training feature matrix (numeric only, pandas DataFrame).
    y_train          : Target series.
    model_save_path        : Path where the trained model (.pkl) will be saved.
    scaler_save_path       : Path to save the fitted scaler (.pkl). Defaults to same dir → 'scaler.pkl'.
    feature_names_save_path: Path to save the feature names list (.pkl). Defaults to same dir → 'feature_names.pkl'.
    n_estimators      : Number of boosting rounds (default 300).
    max_depth         : Maximum tree depth (default 6).
    learning_rate     : Step size shrinkage (default 0.05).
    subsample         : Row sub-sampling ratio per tree (default 0.8).
    colsample_bytree  : Column sub-sampling ratio per tree (default 0.8).
    random_state      : RNG seed for reproducibility (default 42).

    Returns
    -------
    (model, scaler) — fitted XGBRegressor and StandardScaler.
    """
    # Resolve artifact paths
    model_path = Path(model_save_path)
    if scaler_save_path is None:
        scaler_save_path = str(model_path.parent / "scaler.pkl")
    if feature_names_save_path is None:
        feature_names_save_path = str(model_path.parent / "feature_names.pkl")

    # Ensure output directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Capture feature names before converting to numpy
    feature_names = list(X_train.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.values)   # .values → numpy, no column names

    # 2. Train XGBoost
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
        tree_method="hist",        # fast histogram-based algorithm
        verbosity=0,               # suppress XGBoost training logs
    )
    model.fit(X_scaled, y_train.values)


    # 3. Persist all three artifacts
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(feature_names, feature_names_save_path)

    return model, scaler