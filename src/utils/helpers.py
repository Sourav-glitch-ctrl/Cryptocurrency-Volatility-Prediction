"""
helpers.py
----------
General-purpose utility helpers for the crypto-volatility project.
"""

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: str) -> Path:
    """
    Create ``path`` (and any intermediate directories) if it does not exist.

    Parameters
    ----------
    path : Directory path to create.

    Returns
    -------
    pathlib.Path  — the resolved path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_artifact(obj: Any, path: str) -> None:
    """
    Persist any Python object to disk using joblib.

    Parent directories are created automatically.

    Parameters
    ----------
    obj  : Object to serialize (e.g., sklearn model, scaler, array).
    path : Destination file path (.pkl recommended).
    """
    ensure_dir(str(Path(path).parent))
    joblib.dump(obj, path)


def load_artifact(path: str) -> Any:
    """
    Load an object previously saved with :func:`save_artifact`.

    Parameters
    ----------
    path : Path to the serialized file.

    Returns
    -------
    Deserialized Python object.

    Raises
    ------
    FileNotFoundError  — if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    return joblib.load(p)


def get_feature_columns(df: pd.DataFrame, exclude: list = None) -> list:
    """
    Return numeric column names from a DataFrame, optionally excluding some.

    Parameters
    ----------
    df      : Input DataFrame.
    exclude : List of column names to exclude (e.g. ['target']).

    Returns
    -------
    list of str  — selected column names.
    """
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include=["number"]).columns if c not in exclude]
