"""
utils.py  (app/)
----------------
Helper functions for the Streamlit app:
  - Plotly chart builders
  - Model / scaler existence checks
  - Metrics formatting
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------------ #
#  Path helpers
# ------------------------------------------------------------------ #

def model_exists(
    model_path: str = "models/trained_model.pkl",
    scaler_path: str = "models/scaler.pkl",
    feature_names_path: str = "models/feature_names.pkl",
) -> bool:
    """Return True when model, scaler, and feature names files are present on disk."""
    return Path(model_path).exists() and Path(scaler_path).exists() and Path(feature_names_path).exists()


# ------------------------------------------------------------------ #
#  Chart builders
# ------------------------------------------------------------------ #

def plot_price(df: pd.DataFrame) -> go.Figure:
    """Candlestick chart if OHLC present, else line chart of close."""
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        x = df.index if "timestamp" not in df.columns else df["timestamp"]
        fig = go.Figure(
            data=go.Candlestick(
                x=x,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        )
        fig.update_layout(
            title="Price (OHLC)",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
        )
    else:
        fig = px.line(df, y="close", title="Close Price", template="plotly_dark", height=350)
    return fig


def plot_volatility(result: pd.DataFrame) -> go.Figure:
    """Line chart comparing actual vs predicted volatility."""
    fig = go.Figure()

    if "volatility" in result.columns:
        fig.add_trace(go.Scatter(
            y=result["volatility"],
            mode="lines",
            name="Actual Volatility",
            line=dict(color="#60a5fa", width=1.5),
        ))

    if "prediction" in result.columns:
        fig.add_trace(go.Scatter(
            y=result["prediction"],
            mode="lines",
            name="Predicted Volatility",
            line=dict(color="#f59e0b", width=2, dash="dot"),
        ))

    fig.update_layout(
        title="Actual vs Predicted Volatility",
        xaxis_title="Time Step",
        yaxis_title="Volatility",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_feature_importance(model, feature_names: list) -> Optional[go.Figure]:
    """Horizontal bar chart of feature importances (works for RF/GBT)."""
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]  # top 20

    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker_color="#818cf8",
    ))
    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance",
        template="plotly_dark",
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Scatter plot of residuals vs actual values."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig = go.Figure(go.Scatter(
        x=y_true,
        y=residuals,
        mode="markers",
        marker=dict(color="#34d399", size=4, opacity=0.7),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.6)
    fig.update_layout(
        title="Residuals vs Actual",
        xaxis_title="Actual Volatility",
        yaxis_title="Residual (Actual − Predicted)",
        template="plotly_dark",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ------------------------------------------------------------------ #
#  Metrics formatting
# ------------------------------------------------------------------ #

def format_metrics(metrics: dict) -> pd.DataFrame:
    """Convert a metrics dict to a display-ready DataFrame."""
    rows = [
        {"Metric": "RMSE",  "Value": f"{metrics.get('rmse', 0):.8f}",  "Description": "Root Mean Squared Error"},
        {"Metric": "MAE",   "Value": f"{metrics.get('mae', 0):.8f}",   "Description": "Mean Absolute Error"},
        {"Metric": "R²",    "Value": f"{metrics.get('r2', 0):.4f}",    "Description": "Coefficient of Determination"},
        {"Metric": "MAPE",  "Value": f"{metrics.get('mape', 0):.2f}%", "Description": "Mean Absolute Percentage Error"},
    ]
    return pd.DataFrame(rows)
