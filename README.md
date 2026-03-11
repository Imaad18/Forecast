# FinCast Pro

**Institutional-grade time series forecasting dashboard**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-0467DF?style=flat)](https://facebook.github.io/prophet)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-189FDD?style=flat)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-c8a951?style=flat)](LICENSE)

FinCast Pro is a professional forecasting platform built in Python and Streamlit. It runs four independent models — Prophet, ARIMA/SARIMA, XGBoost, and Monte Carlo simulation — on any time series, then blends them into an ensemble forecast. The app includes walk-forward backtesting, residual diagnostics, bias–variance decomposition, regime detection, and full Excel/CSV export. It is designed for analysts, data scientists, and portfolio managers who need production-quality forecasts without writing code.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation--local-setup)
- [Deployment](#deployment)
- [Usage Guide](#usage-guide)
- [Technical Reference](#technical-reference)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

### Data Sources
- **Upload any CSV** — flexible column detection auto-maps `date/time/ds` and `value/close/price/revenue/y` columns; no specific naming required
- **Yahoo Finance integration** — enter any ticker (`AAPL`, `BTC-USD`, `MSFT`, `ETH-USD`, `^GSPC`) and pull up to 10 years of live OHLCV data
- **Built-in sample datasets** — 48-month monthly revenue with seasonality and injected anomaly, plus a 500-day GBM stock simulation

### Pre-Processing
- **Auto frequency detection** — infers Monthly / Quarterly / Weekly / Daily (Business) from the median gap between dates
- **IQR anomaly detection** — flags outliers with a configurable IQR multiplier (1×–4×), overlaid on the historical chart
- **Winsorisation toggle** — clips anomalies to IQR bounds before fitting so models train on clean data; raw data is preserved in all charts and exports
- **Regime detection** — CUSUM structural break detection identifies regime shifts with amber overlays on the historical series

### Forecasting Models

| Model | Algorithm | Confidence Interval |
|---|---|---|
| **Prophet** | Meta's additive decomposition with Fourier seasonality and trend changepoints | 90% posterior uncertainty |
| **ARIMA / SARIMA** | Statsmodels SARIMAX(1,1,1) with automatic seasonal order detection | 90% analytical CI |
| **XGBoost** | Gradient-boosted trees on a 12-lag feature window with recursive multi-step forecasting | Expanding uncertainty bands |
| **Monte Carlo** | Geometric Brownian Motion with scenario-adjusted drift and volatility | P10–P90 percentile fan |
| **Ensemble** | Equal-weight blend of all active model point forecasts | Averaged CI bands |

### Risk & Scenario Analysis
- Five scenario presets: **Stress · Bear · Base · Bull · Upside** adjusting GBM drift (μ) and volatility (σ) multipliers
- **VaR (95%)** and **CVaR / Expected Shortfall** computed from the Monte Carlo terminal distribution
- Full percentile table P1–P99 with % change vs. last observed value
- Probability of loss and probability of gain > 10%
- Monte Carlo fan chart with 100 individual simulated paths overlaid

### Model Validation
- **Walk-forward backtesting** — 5-fold rolling-origin out-of-sample validation; each fold trains on an expanding window and tests on the next held-out period
- **Residual diagnostics** — time-series residual plot and ACF bar chart per model with 95% significance bands
- **Bias² vs. Variance decomposition** — bar chart separating systematic error (underfitting) from variance (overfitting)
- **Model leaderboard** — ranked by in-sample MAPE with 🥇🥈🥉 medals and ✦ Best Fit annotation
- **In-sample fit overlay** — all fitted models overlaid against actual data for visual inspection

### Performance & Caching
- `@st.cache_data` on all four model runners — second run with identical inputs returns instantly from cache
- XGBoost uses `tree_method="hist"` — fastest available tree builder
- Monte Carlo is fully vectorised with NumPy — no Python loop over simulations
- Prophet configured with `uncertainty_samples=200` and `n_changepoints=15` — ~5× faster than defaults
- ARIMA uses L-BFGS optimiser with `maxiter=75` — 3–4× faster than Newton default

### Export
| File | Contents |
|---|---|
| Forecast CSV | All models · lower / upper CI · ensemble |
| Historical CSV | Raw series with anomaly flag column |
| Excel Report (multi-sheet) | Historical · Forecast · Model Accuracy · Walk-Forward · MC Summary |

---

## Project Structure

```
fincast-pro/
├── app.py                  # Main application — all logic and UI
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark theme configuration (no injected CSS)
├── sample_revenue.csv      # 48-month sample dataset with anomaly
└── README.md
```

---

## Installation & Local Setup

### Prerequisites

- Python 3.10 or later
- pip

### 1 — Clone the repository

```bash
git clone https://github.com/Imaad18/fincast-pro.git
cd fincast-pro
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Prophet installation note:** Prophet requires a C++ compiler and PyStan. On most systems `pip install prophet` handles this automatically. On Windows, the easiest path is conda:
> ```bash
> conda install -c conda-forge prophet
> ```

### 4 — Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deployment

### Streamlit Cloud (recommended — free tier available)

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your fork → branch `main` → main file `app.py`
4. Click **Deploy**

Streamlit Cloud reads `requirements.txt` and `.streamlit/config.toml` automatically. No secrets or environment variables are required for any feature.

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

```bash
docker build -t fincast-pro .
docker run -p 8501:8501 fincast-pro
```

---

## Usage Guide

### Using Your Own CSV

Your file needs at minimum two columns — a date column and a numeric value column. Column names are detected automatically:

| Accepted date column names | Accepted value column names |
|---|---|
| `ds`, `date`, `time`, `timestamp`, `period` | `y`, `value`, `close`, `price`, `revenue`, `sales` |

If no column matches either list, the app uses the first two columns and displays the mapping it applied. Minimum recommended series length: 24 observations for monthly data, 8 for quarterly, 104 for weekly, 500 for daily.

### Pulling Live Market Data

1. Select **Yahoo Finance** as the data source in the sidebar
2. Enter a ticker symbol — e.g. `AAPL`, `BTC-USD`, `GLD`, `^GSPC`, `EURUSD=X`, `NVDA`
3. Choose a lookback period: 1y · 2y · 5y · 10y
4. Frequency is auto-detected as Daily (Business); override in the Frequency selector if needed

### Running a Forecast

1. Choose your **Data Source** and load data
2. Confirm or override the **Frequency** (auto-detected for uploads and Yahoo data)
3. Set the **Forecast Horizon** — number of periods ahead
4. Toggle individual **Models** on/off
5. Optionally enable **Winsorise outliers** to clean the training data
6. Choose a **Scenario** for Monte Carlo
7. Toggle **Show confidence intervals** and **Show regime breaks** as desired
8. Click **▶ RUN FORECAST ENGINE**

Each model shows a live status card (Queued → Running → Done with MAPE). Results appear as soon as all models complete. Clicking Run again with unchanged inputs is instant due to caching.

### Interpreting the Validation Tab

- **In-sample MAPE** — how well models fit the training data. Lower is better but can be misleading if the model is overfitting.
- **Walk-forward OOS MAPE** — how well models forecast unseen data across 5 rolling folds. This is the number to trust for deployment decisions.
- **ACF plot** — bars outside the dashed CI lines indicate autocorrelated residuals, meaning the model has not captured all signal.
- **Bias² vs Variance** — dominant red (bias²) means the model is systematically under/over-forecasting; dominant blue (variance) means it is overfitting.

---

## Technical Reference

### Model Parameters

| Model | Parameter | Value | Rationale |
|---|---|---|---|
| Prophet | `yearly_seasonality` | 6 Fourier terms | Faster; sufficient for most business series |
| Prophet | `n_changepoints` | 15 | Reduced from default 25; reduces overfitting |
| Prophet | `uncertainty_samples` | 200 | Reduced from default 1 000; 5× speed improvement |
| Prophet | `changepoint_prior_scale` | 0.05 | Conservative; prevents trend drift |
| ARIMA | Order | (1,1,1) | Robust default for integrated financial series |
| ARIMA | Seasonal order | Auto | Enabled when `len(s) ≥ 3 × seasonal_period` |
| ARIMA | Optimiser | L-BFGS | 3–4× faster than Newton |
| ARIMA | `maxiter` | 75 | Sufficient with L-BFGS |
| XGBoost | `n_estimators` | 150 | Diminishing returns beyond this for typical TS lengths |
| XGBoost | `tree_method` | hist | Fastest available; equivalent accuracy |
| XGBoost | `max_depth` | 3 | Shallower reduces overfitting on small datasets |
| XGBoost | Lag window | 12 | 12 lagged values + rolling mean/std/min/max as features |
| Monte Carlo | Paths | 500–5 000 | User-configurable; vectorised via NumPy |
| Monte Carlo | Model | GBM | Log-normal returns, seed 99 for reproducibility |

### Scenario Parameters (Monte Carlo)

| Scenario | Drift multiplier (μ×) | Volatility multiplier (σ×) |
|---|---|---|
| Stress | 0.3 | 1.5 |
| Bear | 0.3 | 1.5 |
| Base | 1.0 | 1.0 |
| Bull | 1.5 | 0.65 |
| Upside | 1.5 | 0.65 |

### Financial Metrics Computed

| Metric | Formula |
|---|---|
| CAGR | `(last / first) ^ (1 / years) − 1` × 100 |
| Max Drawdown | `min((s − cummax(s)) / cummax(s))` × 100 |
| Sharpe Ratio | `(annualised_return − 0.05) / annualised_std` |
| VaR 95% | 5th percentile of Monte Carlo terminal distribution |
| CVaR / ES | Mean of simulated terminal values ≤ VaR |
| MAPE | `mean(|actual − forecast| / |actual|)` × 100 |
| sMAPE | `mean(2|f − a| / (|a| + |f|))` × 100 |
| RMSE | `sqrt(mean((actual − forecast)²))` |

### Walk-Forward Backtesting Logic

```
min_train  = max(24, n // 3)
step       = max(1, (n − min_train) // 5)
splits     = [(min_train + i×step, min_train + (i+1)×step) for i in 0..4]
```

Each split trains on `df[:train_end]` and forecasts `step` periods. Results are cached per `(data_hash, freq)` so repeated visits to the Validation tab do not re-run.

### Regime Detection Algorithm

```
z          = (s − mean(s)) / std(s)        # Z-score normalisation
cusum      = cumsum(z)                      # Cumulative sum
break if   |cusum[i] − cusum[last_break]| > 2.5
min_gap    = 6 observations between breaks
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.30 | Web application framework |
| `plotly` | ≥ 5.0 | Interactive charts |
| `pandas` | ≥ 2.0 | Data manipulation |
| `numpy` | ≥ 1.24 | Numerical computation |
| `prophet` | ≥ 1.1 | Additive time series model (Meta) |
| `statsmodels` | ≥ 0.14 | ARIMA / SARIMA |
| `xgboost` | ≥ 2.0 | Gradient boosted trees |
| `yfinance` | ≥ 0.2 | Yahoo Finance data fetching |
| `openpyxl` | ≥ 3.1 | Excel export engine |
| `scikit-learn` | ≥ 1.3 | Utility functions |

---

## Roadmap

- [ ] **Multi-series support** — upload a CSV with multiple value columns and forecast each independently
- [ ] **LSTM / Temporal Fusion Transformer** — deep learning baseline for comparison
- [ ] **Auto-ARIMA** — AIC/BIC grid search over (p, d, q) order combinations
- [ ] **API key support via `st.secrets`** — user-provided Alpha Vantage / FRED / Quandl keys for alternative data
- [ ] **PDF report export** — one-click professional report with all charts embedded
- [ ] **Mobile layout optimisation** — responsive sidebar collapse for small screens
- [ ] **Forecast accuracy tracking** — save past runs and compare forecast vs. actuality over time

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/fincast-pro.git
cd fincast-pro

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make changes and commit
git add .
git commit -m "feat: brief description of change"

# 4. Push and open a pull request
git push origin feature/your-feature-name
```

**Commit convention:** `feat:` · `fix:` · `docs:` · `refactor:` · `perf:`

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full terms.

---

## Author

**Imaad**

- GitHub: [@Imaad18](https://github.com/Imaad18)
- Repository: [github.com/Imaad18/fincast-pro](https://github.com/Imaad18/fincast-pro)

---

*Built with Python · Streamlit · Prophet · XGBoost · Plotly · yfinance*


