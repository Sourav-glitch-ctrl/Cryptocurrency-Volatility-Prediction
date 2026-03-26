"""
validation.py
-------------
Validates that a DataFrame meets the minimum requirements for the
crypto-volatility prediction pipeline.

Note: the file is intentionally named `validation.py` (not `validate.py`)
so that `from src.data.validation import validate_data` works correctly.
The old `validate.py` stub is kept alongside this file for compatibility.
"""

import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate a preprocessed DataFrame.

    Checks:
    - DataFrame is not empty.
    - All required OHLC columns are present.
    - No missing values exist.
    - All values in OHLC columns are non-negative.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bool  — True when all checks pass.

    Raises
    ------
    ValueError  — on any validation failure.
    """
    if df is None or len(df) == 0:
        raise ValueError("Validation failed: DataFrame is empty.")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Validation failed: missing required columns: {missing_cols}")

    null_count = df[list(REQUIRED_COLUMNS)].isnull().sum().sum()
    if null_count > 0:
        raise ValueError(
            f"Validation failed: {null_count} missing value(s) found in OHLC columns."
        )

    for col in REQUIRED_COLUMNS:
        if (df[col] < 0).any():
            raise ValueError(f"Validation failed: column '{col}' contains negative values.")

    return True
