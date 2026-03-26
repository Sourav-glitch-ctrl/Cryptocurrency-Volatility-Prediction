"""
volatility.py
-------------
Advanced volatility estimators for crypto price series.
"""

import numpy as np
import pandas as pd


def calculate_realized_volatility(
    df: pd.DataFrame,
    window: int = 10,
    annualize: bool = False,
    trading_periods: int = 365,
) -> pd.DataFrame:
    """
    Rolling realized (standard-deviation) volatility of log returns.

    Parameters
    ----------
    df              : DataFrame with a 'close' column.
    window          : Rolling look-back period (default 10).
    annualize       : If True, multiply by sqrt(trading_periods).
    trading_periods : Used only when annualize=True (default 365 for crypto).

    Returns
    -------
    DataFrame with new column ``realized_vol_{window}``.
    """
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    rv = log_ret.rolling(window).std()

    if annualize:
        rv = rv * np.sqrt(trading_periods)

    df[f"realized_vol_{window}"] = rv
    return df


def calculate_parkinson_volatility(
    df: pd.DataFrame,
    window: int = 10,
    annualize: bool = False,
    trading_periods: int = 365,
) -> pd.DataFrame:
    """
    Parkinson (high-low) volatility estimator.

    Uses only high and low prices; more efficient than close-to-close.

    Parameters
    ----------
    df              : DataFrame with 'high' and 'low' columns.
    window          : Rolling look-back period (default 10).
    annualize       : If True, multiply by sqrt(trading_periods).
    trading_periods : Used only when annualize=True (default 365 for crypto).

    Returns
    -------
    DataFrame with new column ``parkinson_vol_{window}``.
    """
    df = df.copy()
    log_hl = np.log(df["high"] / df["low"])
    pk = np.sqrt(
        (1 / (4 * np.log(2)))
        * log_hl.pow(2).rolling(window).mean()
    )

    if annualize:
        pk = pk * np.sqrt(trading_periods)

    df[f"parkinson_vol_{window}"] = pk
    return df


def calculate_garman_klass_volatility(
    df: pd.DataFrame,
    window: int = 10,
    annualize: bool = False,
    trading_periods: int = 365,
) -> pd.DataFrame:
    """
    Garman-Klass volatility estimator (uses OHLC).

    Parameters
    ----------
    df              : DataFrame with 'open', 'high', 'low', 'close' columns.
    window          : Rolling look-back period (default 10).
    annualize       : If True, multiply by sqrt(trading_periods).
    trading_periods : Used only when annualize=True (default 365 for crypto).

    Returns
    -------
    DataFrame with new column ``garman_klass_vol_{window}``.
    """
    df = df.copy()
    ln_hl = np.log(df["high"] / df["low"]).pow(2)
    ln_co = np.log(df["close"] / df["open"]).pow(2)

    gk = np.sqrt(
        (0.5 * ln_hl - (2 * np.log(2) - 1) * ln_co).rolling(window).mean()
    )

    if annualize:
        gk = gk * np.sqrt(trading_periods)

    df[f"garman_klass_vol_{window}"] = gk
    return df