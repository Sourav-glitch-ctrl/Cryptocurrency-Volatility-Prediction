"""
data_loader.py
--------------
Loads raw CSV data from disk and optionally parses date/timestamp columns.
"""

import pandas as pd
from pathlib import Path


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Automatically detects and parses a 'timestamp' or 'date' column
    as datetime and sets it as the index.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Auto-detect and parse datetime column
    for col in ["timestamp", "date", "Date", "Timestamp", "time", "Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values(by=col).reset_index(drop=True)
            break

    return df