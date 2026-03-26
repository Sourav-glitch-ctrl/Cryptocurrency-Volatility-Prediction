"""
evaluate.py
-----------
Model evaluation utilities that return a full metrics dictionary.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate(y_true, y_pred) -> dict:
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like — actual target values.
    y_pred : array-like — model predictions.

    Returns
    -------
    dict with keys: rmse, mae, r2, mape
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # RMSE — compatible with all sklearn versions (avoids deprecated squared= kwarg)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    # MAPE (guard against division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_vals = np.where(
            y_true != 0,
            np.abs((y_true - y_pred) / y_true) * 100,
            np.nan,
        )
    mape = float(np.nanmean(mape_vals))

    metrics = {
        "rmse": rmse,
        "mae":  mae,
        "r2":   r2,
        "mape": mape,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k.upper():>6}: {v:.6f}")
    print("==========================\n")