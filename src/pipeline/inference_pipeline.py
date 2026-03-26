"""
inference_pipeline.py
---------------------
Loads a trained model + scaler and generates predictions for new data.
"""

import sys
from pathlib import Path

# Allow running as a script from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.predict import load_artifacts, predict
from src.utils.helpers import get_feature_columns
from src.utils.logger import get_logger

logger = get_logger()


def run_inference(
    raw_df: pd.DataFrame,
    model_path: str = "models/trained_model.pkl",
    scaler_path: str = "models/scaler.pkl",
    feature_names_path: str = "models/feature_names.pkl",
) -> pd.DataFrame:
    """
    Run the inference pipeline on a raw input DataFrame.

    Steps:
    1. Preprocess (same transformations as training).
    2. Build features (same transformations as training).
    3. Load model + scaler + feature names from disk.
    4. Scale & predict.
    5. Attach 'prediction' column to the result DataFrame.

    Parameters
    ----------
    raw_df      : Raw OHLCV DataFrame (as uploaded by user or received from API).
    model_path  : Path to the saved model pkl.
    scaler_path : Path to the saved scaler pkl.
    feature_names_path: Path to the saved feature names list pkl.

    Returns
    -------
    pd.DataFrame — feature DataFrame with an added 'prediction' column.
    """
    logger.info("Running inference pipeline...")

    # 1. Preprocess
    df = preprocess_data(raw_df)

    # 2. Feature engineering
    df = build_features(df)

    # 3. Load artifacts
    model, scaler, feature_names = load_artifacts(model_path, scaler_path, feature_names_path)

    # 4. Select feature columns (exclude target if it exists)
    feature_cols = get_feature_columns(df, exclude=["target"])
    X = df[feature_cols]

    # 5. Predict (scaler is applied inside predict())
    preds = predict(model, X, scaler=scaler)
    df["prediction"] = preds

    logger.info("Inference complete. %d predictions generated.", len(preds))
    return df