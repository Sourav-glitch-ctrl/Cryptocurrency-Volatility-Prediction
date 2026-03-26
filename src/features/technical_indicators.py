"""
technical_indicators.py
-----------------------
Pure-function technical indicator calculators.
All functions accept a DataFrame and return it with the new column(s) added.
"""

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple percentage returns column."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    return df


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns column."""
    df = df.copy()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_volatility(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add rolling standard deviation of returns as volatility column."""
    df = df.copy()
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(window).std()
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI).

    Parameters
    ----------
    df     : DataFrame with a 'close' column.
    window : Look-back period (default 14).
    """
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Add Bollinger Bands: middle, upper, lower, and bandwidth.

    Parameters
    ----------
    df      : DataFrame with a 'close' column.
    window  : Rolling window for mean & std (default 20).
    num_std : Number of standard deviations for upper/lower bands (default 2).
    """
    df = df.copy()
    rolling_mean = df["close"].rolling(window).mean()
    rolling_std = df["close"].rolling(window).std()

    df[f"bb_mid_{window}"]   = rolling_mean
    df[f"bb_upper_{window}"] = rolling_mean + num_std * rolling_std
    df[f"bb_lower_{window}"] = rolling_mean - num_std * rolling_std
    df[f"bb_width_{window}"] = (
        df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
    ) / rolling_mean
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Add MACD line, Signal line, and Histogram.

    Parameters
    ----------
    df     : DataFrame with a 'close' column.
    fast   : Fast EMA period (default 12).
    slow   : Slow EMA period (default 26).
    signal : Signal EMA period (default 9).
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    df["macd_line"]      = ema_fast - ema_slow
    df["macd_signal"]    = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
    return df