"""
build_features.py
-----------------
Assembles all features from technical indicators, volatility estimators,
lag features, and rolling statistics.

All features are built using only past data (shifted where needed)
to prevent data leakage.
"""

import pandas as pd

from src.features.technical_indicators import (
    add_returns,
    add_log_returns,
    add_rsi,
    add_bollinger_bands,
    add_macd,
)
from src.features.volatility import (
    calculate_realized_volatility,
    calculate_parkinson_volatility,
    calculate_garman_klass_volatility,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from a preprocessed OHLCV DataFrame.

    Leakage-safe: every feature that uses rolling windows is shifted by 1
    so that at time t the model only sees data up to t-1.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with at minimum 'open', 'high', 'low', 'close'.

    Returns
    -------
    pd.DataFrame
        DataFrame with all features and a 'target' column (next-period realized volatility).
        Rows with remaining NaN values are dropped.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Returns
    # ------------------------------------------------------------------ #
    df = add_returns(df)          # 'returns'
    df = add_log_returns(df)      # 'log_returns'

    # ------------------------------------------------------------------ #
    # 2. Volatility estimates  (realized, parkinson, garman-klass)
    # ------------------------------------------------------------------ #
    df = calculate_realized_volatility(df, window=10)    # realized_vol_10
    df = calculate_realized_volatility(df, window=20)    # realized_vol_20
    df = calculate_parkinson_volatility(df, window=10)   # parkinson_vol_10
    df = calculate_garman_klass_volatility(df, window=10)  # garman_klass_vol_10

    # Convenience alias used as reference for the target
    df["volatility"] = df["realized_vol_10"]

    # ------------------------------------------------------------------ #
    # 3. Target: next-period realized volatility (1-step ahead)
    # ------------------------------------------------------------------ #
    df["target"] = df["volatility"].shift(-1)

    # ------------------------------------------------------------------ #
    # 4. Lag features  (shifted so leakage-free)
    # ------------------------------------------------------------------ #
    for lag in [1, 2, 3, 5]:
        df[f"vol_lag{lag}"]  = df["volatility"].shift(lag)
        df[f"ret_lag{lag}"]  = df["returns"].shift(lag)

    # ------------------------------------------------------------------ #
    # 5. Rolling statistics  (all shifted by 1 to avoid leakage)
    # ------------------------------------------------------------------ #
    df["vol_5"]  = df["returns"].rolling(5).std().shift(1)
    df["vol_10"] = df["returns"].rolling(10).std().shift(1)
    df["vol_20"] = df["returns"].rolling(20).std().shift(1)

    # ------------------------------------------------------------------ #
    # 6. Technical indicators  (always reference past candles → safe)
    # ------------------------------------------------------------------ #
    df = add_rsi(df, window=14)             # rsi_14
    df = add_bollinger_bands(df, window=20) # bb_mid_20, bb_upper_20, bb_lower_20, bb_width_20
    df = add_macd(df)                       # macd_line, macd_signal, macd_histogram

    # ------------------------------------------------------------------ #
    # 7. Price-derived features  (current candle — no future info)
    # ------------------------------------------------------------------ #
    df["hl_diff"]    = df["high"] - df["low"]           # intra-bar range
    df["co_diff"]    = df["close"] - df["open"]         # body size
    df["hl_ratio"]   = df["hl_diff"] / df["close"]      # normalized range

    # Moving averages (shifted)
    df["ma_5"]   = df["close"].rolling(5).mean().shift(1)
    df["ma_10"]  = df["close"].rolling(10).mean().shift(1)
    df["ma_20"]  = df["close"].rolling(20).mean().shift(1)
    df["ma_ratio"] = df["ma_5"] / df["ma_20"]

    # Momentum
    df["momentum_5"]  = df["close"].shift(1) - df["close"].shift(6)
    df["momentum_10"] = df["close"].shift(1) - df["close"].shift(11)

    # Vol of vol
    df["vol_of_vol"] = df["volatility"].rolling(5).std().shift(1)

    # ------------------------------------------------------------------ #
    # 8. Drop any remaining non-numeric columns and NaN rows
    # ------------------------------------------------------------------ #
    non_num = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_num:
        df.drop(columns=non_num, inplace=True)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df