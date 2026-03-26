# Crypto Volatility Predictor

Forecasts next-period realized volatility for crypto OHLCV data using **XGBoost** and a **Streamlit** dashboard.

---

## Tech Stack

- **Model:** XGBoost + StandardScaler
- **Dashboard:** Streamlit
- **Features:** RSI, MACD, Bollinger Bands, Realized Vol, Parkinson Vol, Garman-Klass Vol, lags
- **Data:** pandas, numpy | **Charts:** Plotly

---

## Project Structure

```
├── app/
│   ├── app.py              ← Streamlit dashboard
│   └── utils.py            ← Chart helpers
├── src/
│   ├── data/               ← loader, preprocess, validation
│   ├── features/           ← build_features, indicators, volatility
│   ├── models/             ← train, predict, evaluate
│   ├── utils/              ← logger, config, helpers
│   └── pipeline/           ← train_pipeline, inference_pipeline
├── config/config.yaml      ← All settings
├── data/raw/               ← Put your CSV here
├── models/                 ← Saved artifacts (auto-generated)
└── requirements.txt
```

---

## Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/crypto-volatility-predictor.git
cd crypto-volatility-predictor
```

### 2. Create a virtual environment

```bash
python -m venv crypto
crypto\Scripts\activate        # Windows
# source crypto/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your data

Place your OHLCV CSV at `data/raw/crypto_data.csv`.

Required columns: `open`, `high`, `low`, `close` (plus optional `volume`, `timestamp`)

### 5. Run the app

```bash
streamlit run app/app.py
```

Open **http://localhost:8501** — use the **Train** tab first, then **Predict**.

---

## App Tabs

| Tab | What it does |
|-----|-------------|
| 🔮 Predict | Upload CSV → run inference → charts + download |
| 🏋️ Train | Upload CSV → tune hyperparams → retrain model |
| 📊 Model Info | Feature importance chart |
| ℹ️ About | Project description |

---

> ⚠️ For educational purposes only. Not financial advice.
