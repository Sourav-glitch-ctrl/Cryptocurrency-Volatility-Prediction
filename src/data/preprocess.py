"""
preprocess.py
-------------
Cleans and prepares raw OHLCV data for feature engineering.
"""

import pandas as pd


# Columns that should not be fed into the model as numeric features
_NON_NUMERIC_COLS = {"timestamp", "date", "Date", "Timestamp", "time", "Time", "symbol", "Symbol"}


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw OHLCV DataFrame:

    1. Remove duplicate rows.
    2. Sort by timestamp / date if present.
    3. Convert timestamp column to datetime, then drop it (avoids
       str → float errors when the DataFrame enters sklearn).
    4. Forward-fill then drop any remaining NaN rows.
    5. Ensure OHLCV columns are numeric (coerce bad values to NaN, then drop).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  — clean, numeric-only (except the target later)
    """
    df = df.copy()

    # 1. Remove duplicates
    df.drop_duplicates(inplace=True)

    # 2. Sort by time column if present, then drop it
    for col in _NON_NUMERIC_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df = df.sort_values(by=col)
            except Exception:
                pass
            df.drop(columns=[col], inplace=True)

    # 3. Force OHLCV columns to numeric; coerce anything unrecognisable to NaN
    ohlcv_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    for col in ohlcv_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Drop any remaining non-numeric columns (e.g. leftover string cols)
    non_num = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_num:
        df.drop(columns=non_num, inplace=True)

    # 5. Forward-fill then drop rows still containing NaN
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df