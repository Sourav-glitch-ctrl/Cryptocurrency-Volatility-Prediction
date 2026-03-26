"""
train_pipeline.py
-----------------
End-to-end training pipeline:
  load → preprocess → validate → build features → split → train → evaluate → save
"""

import sys
from pathlib import Path

# Allow running as a script from the project root (python -m src.pipeline.train_pipeline)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_raw_data
from src.data.preprocess import preprocess_data
from src.data.validation import validate_data          # fixed: was 'validate'

from src.features.build_features import build_features

from src.models.train import train_model
from src.models.predict import predict
from src.models.evaluate import evaluate, print_metrics

from src.utils.logger import get_logger
from src.utils.helpers import ensure_dir, get_feature_columns

logger = get_logger()


def run_training_pipeline(
    data_path: str = "data/raw/crypto_data.csv",
    model_save_path: str = "models/trained_model.pkl",
    scaler_save_path: str = "models/scaler.pkl",
    feature_names_save_path: str = "models/feature_names.pkl",
) -> tuple:
    """
    Run the complete training pipeline.

    Parameters
    ----------
    data_path       : Path to the raw CSV data file.
    model_save_path : Where to save the trained model (.pkl).
    scaler_save_path: Where to save the fitted scaler (.pkl).
    feature_names_save_path: Where to save the feature names list (.pkl).

    Returns
    -------
    (model, scaler, metrics_dict)
    """
    # Ensure output directory exists
    ensure_dir(str(Path(model_save_path).parent))

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info("Loading data from: %s", data_path)
    df = load_raw_data(data_path)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    logger.info("Preprocessing...")
    df = preprocess_data(df)
    logger.info("After preprocessing: %d rows.", len(df))

    # ------------------------------------------------------------------
    # 3. Validate
    # ------------------------------------------------------------------
    logger.info("Validating...")
    validate_data(df)
    logger.info("Validation passed.")

    # ------------------------------------------------------------------
    # 4. Build features
    # ------------------------------------------------------------------
    logger.info("Building features...")
    df = build_features(df)
    logger.info("Feature matrix shape: %s", df.shape)

    # ------------------------------------------------------------------
    # 5. Time-series split (no shuffle — preserve temporal order)
    # ------------------------------------------------------------------
    logger.info("Splitting data (80/20 time-series split)...")
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    feature_cols = get_feature_columns(df, exclude=["target"])

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["target"]

    logger.info(
        "Train size: %d | Test size: %d | Features: %d",
        len(X_train), len(X_test), len(feature_cols),
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("Training XGBRegressor...")
    model, scaler = train_model(
        X_train, y_train,
        model_save_path=model_save_path,
        scaler_save_path=scaler_save_path,
        feature_names_save_path=feature_names_save_path,
    )
    logger.info("Model saved to '%s'.", model_save_path)
    logger.info("Scaler saved to '%s'.", scaler_save_path)

    # ------------------------------------------------------------------
    # 7. Predict & Evaluate
    # ------------------------------------------------------------------
    logger.info("Predicting on test set...")
    preds = predict(model, X_test, scaler=scaler)

    logger.info("Evaluating...")
    metrics = evaluate(y_test, preds)
    print_metrics(metrics)

    return model, scaler, metrics


if __name__ == "__main__":
    run_training_pipeline()