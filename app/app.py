"""
app.py — Crypto Volatility Prediction  (Streamlit)
===================================================
Multi-tab dashboard:
  Tab 1 – Predict        → upload CSV → run inference → charts + download
  Tab 2 – Train          → retrain model on uploaded data
  Tab 3 – Model Info     → feature importance
  Tab 4 – About          → project description
"""

import sys
import os
from pathlib import Path

# ── Make project root importable when launched from any CWD ──────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# App utilities (same directory)
from utils import (
    model_exists,
    plot_price,
    plot_volatility,
    plot_feature_importance,
    plot_residuals,
    format_metrics,
)

# Pipeline
from src.pipeline.inference_pipeline import run_inference
from src.pipeline.train_pipeline import run_training_pipeline
from src.models.evaluate import evaluate
from src.models.predict import load_artifacts
from src.utils.helpers import get_feature_columns


# ────────────────────────────────────────────────────────────────── #
#  Page configuration
# ────────────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────── #
#  Custom CSS
# ────────────────────────────────────────────────────────────────── #
st.markdown("""
<style>
/* ---------- Global ---------- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.85);
    border-right: 1px solid rgba(99,102,241,0.3);
}

/* ---------- Metric cards ---------- */
.metric-card {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.3);
}
.metric-label  { font-size: 0.78rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value  { font-size: 1.7rem; font-weight: 700; color: #818cf8; }
.metric-desc   { font-size: 0.7rem;  color: #64748b; margin-top: 2px; }

/* ---------- Section header ---------- */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a5b4fc;
    border-bottom: 1px solid rgba(165,180,252,0.25);
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ---------- Pills ---------- */
.pill-green { display:inline-block; background:#065f46; color:#6ee7b7; border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; }
.pill-red   { display:inline-block; background:#7f1d1d; color:#fca5a5; border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; }

/* ---------- Streamlit overrides ---------- */
h1, h2, h3 { color: #e2e8f0; }
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 10px; }
.stTabs [data-baseweb="tab"]      { color: #94a3b8; }
.stTabs [aria-selected="true"]    { color: #818cf8 !important; font-weight: 700; }
div[data-testid="stDataFrame"]    { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────── #
#  Sidebar
# ────────────────────────────────────────────────────────────────── #
with st.sidebar:
    st.markdown("## ⚡ Crypto Volatility")
    st.markdown("---")

    MODEL_PATH  = st.text_input("Model path",  value="models/trained_model.pkl")
    SCALER_PATH = st.text_input("Scaler path", value="models/scaler.pkl")

    FEATURE_NAMES_PATH = st.text_input("Feature names path", value="models/feature_names.pkl")

    ready = model_exists(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
    if ready:
        st.markdown('<span class="pill-green">✓ Model ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-red">✗ No model found — train first</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Required CSV columns:**
    - `open`, `high`, `low`, `close`
    - `volume` *(optional)*
    - `timestamp` / `date` *(optional)*
    """)

# ────────────────────────────────────────────────────────────────── #
#  Title
# ────────────────────────────────────────────────────────────────── #
st.markdown("# 📈 Crypto Volatility Predictor")
st.markdown("*Upload OHLCV data → get next-period volatility forecasts powered by RandomForest.*")
st.markdown("---")

# ────────────────────────────────────────────────────────────────── #
#  Tabs
# ────────────────────────────────────────────────────────────────── #
tab_predict, tab_train, tab_info, tab_about = st.tabs([
    "🔮  Predict", "🏋️  Train", "📊  Model Info", "ℹ️  About"
])


# ══════════════════════════════════════════════════════════════════ #
#  TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════ #
with tab_predict:
    st.markdown('<div class="section-header">Upload Data & Run Inference</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a CSV file (OHLC data)",
        type=["csv"],
        key="predict_upload",
        help="Must contain open, high, low, close columns.",
    )

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)

        # ── Raw data preview ───────────────────────────────────────
        with st.expander("📄 Raw Data Preview", expanded=False):
            st.dataframe(raw_df.head(50), use_container_width=True)
            st.caption(f"{len(raw_df):,} rows × {len(raw_df.columns)} columns")

        # ── Price chart ────────────────────────────────────────────
        st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_price(raw_df), use_container_width=True)

        # ── Run inference ──────────────────────────────────────────
        if not model_exists(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH):
            st.warning("⚠️ No trained model found. Please go to the **Train** tab first.")
        else:
            with st.spinner("Running inference pipeline…"):
                try:
                    result = run_inference(raw_df, MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    st.stop()

            st.success(f"✅ Predictions generated for **{len(result):,}** rows.")

            # ── Volatility chart ───────────────────────────────────
            st.markdown('<div class="section-header">Predicted Volatility</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_volatility(result), use_container_width=True)

            # ── Metrics (if target available) ──────────────────────
            if "target" in result.columns:
                st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)
                valid = result.dropna(subset=["target", "prediction"])
                if len(valid) > 0:
                    metrics = evaluate(valid["target"], valid["prediction"])

                    c1, c2, c3, c4 = st.columns(4)
                    cards = [
                        (c1, "RMSE",  f"{metrics['rmse']:.6f}", "Root Mean Sq. Error"),
                        (c2, "MAE",   f"{metrics['mae']:.6f}",  "Mean Abs. Error"),
                        (c3, "R²",    f"{metrics['r2']:.4f}",   "Coeff. of Determination"),
                        (c4, "MAPE",  f"{metrics['mape']:.2f}%","Mean Abs. % Error"),
                    ]
                    for col, label, val, desc in cards:
                        with col:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{val}</div>
                                <div class="metric-desc">{desc}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Residual plot
                    st.markdown('<div class="section-header">Residual Analysis</div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        plot_residuals(valid["target"], valid["prediction"].values),
                        use_container_width=True,
                    )

            # ── Predictions table ──────────────────────────────────
            with st.expander("📋 Predictions Table", expanded=False):
                display_cols = [c for c in ["volatility", "target", "prediction"] if c in result.columns]
                st.dataframe(result[display_cols].tail(100), use_container_width=True)

            # ── Download ───────────────────────────────────────────
            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.info("👆 Upload a CSV file to begin. Required columns: `open`, `high`, `low`, `close`.")


# ══════════════════════════════════════════════════════════════════ #
#  TAB 2 — TRAIN
# ══════════════════════════════════════════════════════════════════ #
with tab_train:
    st.markdown('<div class="section-header">Retrain Model</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        train_file = st.file_uploader(
            "Upload training CSV (OHLC data)",
            type=["csv"],
            key="train_upload",
        )

    with col_right:
        st.markdown("**XGBoost Hyperparameters**")
        n_estimators   = st.slider("n_estimators (rounds)", 50, 600, 300, step=50)
        learning_rate  = st.slider("learning_rate", 0.01, 0.30, 0.05, step=0.01, format="%.2f")
        max_depth      = st.slider("max_depth", 3, 12, 6)
        test_size      = st.slider("Test split %", 10, 40, 20, step=5)

    if train_file is not None:
        train_raw = pd.read_csv(train_file)
        st.info(f"Loaded **{len(train_raw):,}** rows for training.")

        # Save to disk for the pipeline
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        train_raw.to_csv("data/raw/crypto_data.csv", index=False)

        if st.button("🚀 Start Training", use_container_width=False, type="primary"):
            progress = st.progress(0, text="Initialising…")
            log_box  = st.empty()
            logs: list[str] = []

            def update_log(msg: str, pct: int):
                logs.append(msg)
                log_box.code("\n".join(logs[-8:]))
                progress.progress(pct, text=msg)

            try:
                update_log("Loading & preprocessing data…", 10)
                update_log("Building features…", 30)
                update_log("Training XGBoostRegressor…", 50)

                model, scaler, metrics = run_training_pipeline(
                    data_path=f"data/raw/crypto_data.csv",
                    model_save_path=MODEL_PATH,
                    scaler_save_path=SCALER_PATH,
                    feature_names_save_path=FEATURE_NAMES_PATH,
                )

                update_log("Evaluating…", 90)
                update_log("✅ Training complete!", 100)

                st.success("Model trained and saved successfully!")
                st.balloons()

                # Display metrics
                m1, m2, m3, m4 = st.columns(4)
                cards = [
                    (m1, "RMSE",  f"{metrics['rmse']:.8f}"),
                    (m2, "MAE",   f"{metrics['mae']:.8f}"),
                    (m3, "R²",    f"{metrics['r2']:.4f}"),
                    (m4, "MAPE",  f"{metrics['mape']:.2f}%"),
                ]
                for col, label, val in cards:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{val}</div>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Training failed: {e}")
    else:
        st.info("👆 Upload a CSV file to retrain the model. The same format as the Predict tab is required.")


# ══════════════════════════════════════════════════════════════════ #
#  TAB 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════════ #
with tab_info:
    st.markdown('<div class="section-header">Model Details & Feature Importance</div>',
                unsafe_allow_html=True)

    if not model_exists(MODEL_PATH, SCALER_PATH):
        st.warning("No trained model found. Train a model first.")
    else:
        try:
            model, scaler, feature_names = load_artifacts(MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH)

            # ── Model metadata ─────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Type</div>
                    <div class="metric-value" style="font-size:1rem;">XGBoost</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                n_trees = getattr(model, "n_estimators", "—")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Trees</div>
                    <div class="metric-value">{n_trees}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                n_feat = getattr(model, "n_features_in_", "—")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Input Features</div>
                    <div class="metric-value">{n_feat}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Feature importance chart ───────────────────────────
            if hasattr(model, "feature_importances_") and feature_names is not None:
                fig = plot_feature_importance(model, feature_names)
                if fig:
                    st.plotly_chart(fig, use_container_width=False)
            else:
                st.info("Feature names not available — train the model via the Train tab to see importances.")

        except Exception as e:
            st.error(f"Could not load model: {e}")


# ══════════════════════════════════════════════════════════════════ #
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════ #
with tab_about:
    st.markdown("""
    ## 📈 Crypto Volatility Prediction

    A machine-learning pipeline that forecasts **next-period realized volatility**
    for cryptocurrency price series.

    ---

    ### 🧪 Methodology

    | Step | Description |
    |------|-------------|
    | **Data** | OHLCV CSV input with automatic timestamp parsing |
    | **Preprocessing** | De-duplication, sorting, numeric coercion, forward-fill |
    | **Validation** | OHLC column checks, non-negative constraint |
    | **Features** | Realized vol, Parkinson vol, Garman-Klass vol, RSI, MACD, Bollinger Bands, lag features, rolling stats |
    | **Target** | 1-step-ahead realized volatility |
    | **Model** | **XGBoostRegressor** with StandardScaler |
    | **Evaluation** | RMSE, MAE, R², MAPE on held-out 20% time-series split |

    ---

    ### 🏗️ Project Structure

    ```
    src/
    ├── data/         data_loader, preprocess, validation
    ├── features/     build_features, technical_indicators, volatility
    ├── models/       train, predict, evaluate
    ├── utils/        logger, config, helpers
    └── pipeline/     train_pipeline, inference_pipeline
    app/
    ├── app.py        (this file)
    └── utils.py      Plotly chart builders & metric formatters
    ```

    ---

    ### ⚠️ Disclaimer
    This tool is for **educational and research purposes only**.
    Volatility predictions should not be used as financial advice.
    """)