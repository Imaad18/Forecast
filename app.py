"""
FinCast Pro — Institutional Financial Forecasting Platform
Designed for FP&A Teams & Finance Analysts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import copy
from datetime import datetime

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinCast Pro | Institutional Forecasting",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Institutional Dark Theme
# Palette: Deep slate + warm gold + cream text
# Fonts: Playfair Display · IBM Plex Mono · Syne
# ═══════════════════════════════════════════════════════════════
# Inject Google Fonts via link tag (avoids CSP blocks on Streamlit Cloud)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Mono:wght@300;400;500&family=Syne:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>

:root {
  --bg-0:        #09090e;
  --bg-1:        #0e0f17;
  --bg-2:        #13141e;
  --bg-3:        #191a26;
  --border-dim:  #1e2030;
  --border:      #252840;
  --gold:        #c8a951;
  --gold-dim:    rgba(200,169,81,0.12);
  --gold-glow:   rgba(200,169,81,0.30);
  --cream:       #e8e2d5;
  --cream-dim:   #9e9888;
  --cream-muted: #4a4a5e;
  --green:       #3ecf8e;
  --green-dim:   rgba(62,207,142,0.12);
  --red:         #f25f5c;
  --red-dim:     rgba(242,95,92,0.12);
  --blue:        #5b8dee;
  --blue-dim:    rgba(91,141,238,0.12);
  --amber:       #f0a500;
  --shadow:      0 4px 24px rgba(0,0,0,0.5);
}

html, body { background-color: var(--bg-0) !important; }
.stApp { background-color: var(--bg-0) !important; font-family: 'Syne', sans-serif !important; color: var(--cream) !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

section[data-testid="stSidebar"] {
  background: var(--bg-1) !important;
  border-right: 1px solid var(--border-dim) !important;
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: var(--cream) !important; }

.main .block-container { padding: 0 2.5rem 4rem !important; max-width: 1600px !important; }

.fincast-topbar {
  background: var(--bg-1);
  border-bottom: 1px solid var(--border-dim);
  padding: 0.85rem 2.5rem;
  margin: 0 -2.5rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.fincast-logo-mark {
  font-family: 'Playfair Display', serif;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--cream);
}
.fincast-logo-mark span { color: var(--gold); }
.fincast-logo-tag {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cream-dim);
  border-left: 1px solid var(--border);
  padding-left: 0.6rem;
  margin-left: 0.75rem;
}
.fincast-meta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem;
  color: var(--cream-muted);
  text-align: right;
  line-height: 1.6;
}

.sidebar-brand {
  background: linear-gradient(135deg, var(--bg-2), var(--bg-3));
  border-bottom: 1px solid var(--border-dim);
  padding: 1.25rem 1.5rem 1rem;
  margin-bottom: 0.5rem;
}
.sidebar-brand-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--cream);
}
.sidebar-brand-title em { color: var(--gold); font-style: normal; }
.sidebar-brand-sub {
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--cream-muted);
  margin-top: 0.15rem;
}

.sb-section {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--gold);
  padding: 1rem 1.5rem 0.3rem;
  display: block;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(5,1fr);
  gap: 1px;
  background: var(--border-dim);
  border: 1px solid var(--border-dim);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 2rem;
}
.kpi-cell {
  background: var(--bg-1);
  padding: 1.1rem 1.25rem 1rem;
  position: relative;
  transition: background 0.2s;
}
.kpi-cell:hover { background: var(--bg-2); }
.kpi-cell.accent::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: var(--gold);
}
.kpi-label {
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cream-muted);
  margin-bottom: 0.5rem;
}
.kpi-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.55rem;
  font-weight: 400;
  color: var(--cream);
  line-height: 1;
  letter-spacing: -0.02em;
}
.kpi-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; margin-top: 0.35rem; }
.pos { color: var(--green); }
.neg { color: var(--red); }
.neu { color: var(--cream-dim); }
.warn { color: var(--amber); }

.sec-div { display: flex; align-items: center; gap: 0.75rem; margin: 2rem 0 1rem; }
.sec-div-label {
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.18em;
  text-transform: uppercase; color: var(--gold); white-space: nowrap;
}
.sec-div-line { flex: 1; height: 1px; background: var(--border-dim); }

.panel {
  background: var(--bg-1);
  border: 1px solid var(--border-dim);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}

.analyst-note {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--gold);
  border-radius: 6px;
  padding: 0.9rem 1.2rem;
  margin: 0.75rem 0;
  font-size: 0.83rem;
  color: var(--cream-dim);
  line-height: 1.7;
}
.analyst-note strong { color: var(--cream); }
.note-head {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--gold); margin-bottom: 0.4rem;
}

.signal-box {
  border-radius: 6px;
  padding: 0.75rem 1rem;
  margin: 0.4rem 0;
  font-size: 0.82rem;
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
}
.signal-box.bullish { background: var(--green-dim); border: 1px solid rgba(62,207,142,0.2); }
.signal-box.bearish { background: var(--red-dim);   border: 1px solid rgba(242,95,92,0.2); }
.signal-box.neutral { background: var(--bg-2);      border: 1px solid var(--border); }
.signal-box.caution { background: rgba(240,165,0,0.08); border: 1px solid rgba(240,165,0,0.2); }
.signal-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.signal-text { color: var(--cream-dim); line-height: 1.55; }
.signal-text strong { color: var(--cream); }

.anom-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.5rem 0.9rem; border-radius: 4px;
  background: rgba(242,95,92,0.07); border-left: 2px solid var(--red);
  margin: 0.2rem 0;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.77rem;
}
.anom-date { color: var(--cream-muted); }
.anom-val  { color: var(--cream); }
.anom-dev  { color: var(--red); font-weight: 500; }

.lb-table { width: 100%; border-collapse: collapse; }
.lb-table th {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--cream-muted);
  padding: 0.6rem 1rem; text-align: left;
  border-bottom: 1px solid var(--border-dim); background: var(--bg-2);
}
.lb-table td {
  padding: 0.65rem 1rem; border-bottom: 1px solid var(--border-dim);
  font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--cream);
}
.lb-table tr:last-child td { border-bottom: none; }
.lb-table tr.lb-winner td { background: rgba(200,169,81,0.06); }
.lb-table tr:hover td { background: var(--bg-2); }
.lb-rank {
  display: inline-flex; align-items: center; justify-content: center;
  width: 20px; height: 20px; border-radius: 50%;
  font-size: 0.65rem; font-weight: 700; margin-right: 0.5rem;
  background: var(--border); color: var(--cream-dim);
}
.lb-rank.gold   { background: var(--gold); color: #000; }
.lb-rank.silver { background: #8a8a9a; color: #000; }
.lb-rank.bronze { background: #8b6914; color: #fff; }
.model-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 0.5rem; }

.stat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.45rem 0; border-bottom: 1px solid var(--border-dim); font-size: 0.82rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: var(--cream-muted); }
.stat-val { font-family: 'IBM Plex Mono', monospace; color: var(--cream); }

.export-card {
  background: var(--bg-1); border: 1px solid var(--border-dim);
  border-radius: 8px; padding: 1.25rem 1.5rem; height: 100%;
}
.export-card-title { font-size: 0.78rem; font-weight: 700; color: var(--cream); margin-bottom: 0.35rem; }
.export-card-desc  { font-size: 0.76rem; color: var(--cream-muted); line-height: 1.55; margin-bottom: 1rem; }

.badge {
  display: inline-block; padding: 1px 9px; border-radius: 3px;
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-gold    { background: var(--gold-dim); color: var(--gold); border: 1px solid var(--gold-glow); }
.badge-green   { background: var(--green-dim); color: var(--green); border: 1px solid rgba(62,207,142,0.25); }
.badge-neutral { background: var(--bg-3); color: var(--cream-dim); border: 1px solid var(--border); }

.empty-state { text-align: center; padding: 5rem 2rem; }
.empty-state-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.3; }
.empty-state-msg  { font-size: 0.9rem; color: var(--cream-muted); }
.empty-state-sub  { font-size: 0.76rem; color: var(--cream-muted); opacity: 0.5; margin-top: 0.4rem; }

div.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-1) !important; border-bottom: 1px solid var(--border-dim) !important;
  gap: 0 !important; padding: 0 !important;
}
div.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--cream-muted) !important;
  font-family: 'Syne', sans-serif !important; font-weight: 500 !important;
  font-size: 0.78rem !important; letter-spacing: 0.05em !important;
  padding: 0.75rem 1.5rem !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
}
div.stTabs [aria-selected="true"] {
  color: var(--gold) !important; border-bottom: 2px solid var(--gold) !important;
  background: rgba(200,169,81,0.05) !important;
}
div.stTabs [data-baseweb="tab"]:hover { color: var(--cream) !important; background: var(--bg-2) !important; }

.stButton > button {
  background: var(--gold) !important; color: #0a0a0f !important;
  font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
  font-size: 0.78rem !important; letter-spacing: 0.1em !important;
  text-transform: uppercase !important; border: none !important;
  border-radius: 4px !important; padding: 0.6rem 1.5rem !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
  background: var(--bg-2) !important; border: 1px solid var(--border) !important;
  border-radius: 4px !important; color: var(--cream) !important; font-size: 0.82rem !important;
}

div[data-testid="stFileUploader"] {
  background: var(--bg-2) !important; border: 1px dashed var(--border) !important; border-radius: 6px !important;
}
.stCheckbox label { font-size: 0.8rem !important; color: var(--cream-dim) !important; }

div[data-testid="stDownloadButton"] > button {
  background: transparent !important; color: var(--gold) !important;
  border: 1px solid var(--gold-glow) !important;
  font-size: 0.76rem !important; letter-spacing: 0.08em !important;
  padding: 0.45rem 1rem !important; width: auto !important;
}
div[data-testid="stDownloadButton"] > button:hover { background: var(--gold-dim) !important; opacity: 1 !important; }

#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border-dim) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CHART CONFIG
# ═══════════════════════════════════════════════════════════════

_BASE_LAYOUT = dict(
    paper_bgcolor="#09090e", plot_bgcolor="#09090e",
    font=dict(family="Syne", color="#4a4a5e", size=11),
    xaxis=dict(
        gridcolor="#13141e", gridwidth=1, linecolor="#1e2030", zeroline=False,
        tickfont=dict(color="#4a4a5e", size=10, family="IBM Plex Mono"),
        showspikes=True, spikecolor="#252840", spikethickness=1, spikedash="solid",
    ),
    yaxis=dict(
        gridcolor="#13141e", gridwidth=1, linecolor="#1e2030", zeroline=False,
        tickfont=dict(color="#4a4a5e", size=10, family="IBM Plex Mono"),
    ),
    legend=dict(
        bgcolor="rgba(14,15,23,0.9)", bordercolor="#252840", borderwidth=1,
        font=dict(color="#9e9888", size=10, family="Syne"), itemsizing="constant",
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="#13141e", bordercolor="#252840",
        font=dict(color="#e8e2d5", size=11, family="IBM Plex Mono"), namelength=-1,
    ),
    dragmode="zoom",
)

def cly(**extra):
    lyt = copy.deepcopy(_BASE_LAYOUT)
    lyt.update(extra)
    return lyt

C = {
    "actual":   "#6b6b82",
    "prophet":  "#c8a951",
    "arima":    "#5b8dee",
    "xgboost":  "#3ecf8e",
    "montecarlo":"#f25f5c",
    "ensemble": "#d4a0ff",
}

def model_color(name):
    return C.get(name.lower().replace(" ",""), "#9e9888")

# ═══════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═══════════════════════════════════════════════════════════════

def gen_revenue(n=48):
    np.random.seed(42)
    dates  = pd.date_range(end=datetime.today().replace(day=1), periods=n, freq="MS")
    trend  = np.linspace(480_000, 880_000, n)
    season = 75_000 * np.sin(2*np.pi*np.arange(n)/12 - np.pi/3)
    noise  = np.random.normal(0, 22_000, n)
    vals   = np.clip(trend+season+noise, 0, None)
    vals[n-4] *= 0.14   # inject anomaly
    return pd.DataFrame({"ds": dates, "y": vals.round(0)})

def gen_stock(n=500):
    np.random.seed(7)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    shocks = np.random.normal(0.0003, 0.015, n)
    price  = 120 * np.exp(np.cumsum(shocks))
    return pd.DataFrame({"ds": dates, "y": price.round(4)})

def detect_anomalies(series, factor=2.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - factor*iqr, q3 + factor*iqr
    return (series < lo) | (series > hi), lo, hi

def cagr(series, freq):
    ppy = {"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    n   = len(series)/ppy
    if n <= 0 or series.iloc[0] <= 0: return 0.0
    return ((series.iloc[-1]/series.iloc[0])**(1/n)-1)*100

def max_drawdown(series):
    dd = (series - series.cummax()) / series.cummax()
    return dd.min()*100

def sharpe(series, freq, rfr=0.05):
    ppy  = {"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    rets = series.pct_change().dropna()
    if rets.std() == 0: return 0.0
    return (rets.mean()*ppy - rfr) / (rets.std()*np.sqrt(ppy))

def mape(a,f):
    a,f = np.array(a), np.array(f)
    m   = (a!=0)&~np.isnan(a)&~np.isnan(f)
    return 100*np.mean(np.abs((a[m]-f[m])/a[m]))

def smape(a,f):
    a,f = np.array(a), np.array(f)
    m   = ~(np.isnan(a)|np.isnan(f))
    return 100*np.mean(2*np.abs(f[m]-a[m])/(np.abs(a[m])+np.abs(f[m])+1e-9))

def rmse(a,f):
    a,f = np.array(a), np.array(f)
    m   = ~(np.isnan(a)|np.isnan(f))
    return np.sqrt(np.mean((a[m]-f[m])**2))

# ═══════════════════════════════════════════════════════════════
# MODEL RUNNERS
# ═══════════════════════════════════════════════════════════════

def _offset(freq):
    try:
        return pd.tseries.frequencies.to_offset(freq)
    except Exception:
        return pd.DateOffset(months=1)

def run_prophet(df, horizon, freq):
    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.90,
                    changepoint_prior_scale=0.05)
        m.fit(df[["ds","y"]])
        future = m.make_future_dataframe(periods=horizon, freq=freq)
        fc = m.predict(future)
        fwd = fc[fc["ds"]>df["ds"].max()][["ds","yhat","yhat_lower","yhat_upper"]].copy()
        fwd.columns = ["ds","yhat","lower","upper"]
        ins = fc[fc["ds"]<=df["ds"].max()][["ds","yhat"]].rename(columns={"yhat":"y_pred"})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception:
        return None, None

def run_arima(df, horizon, freq):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        series = df.set_index("ds")["y"]
        s      = {"MS":12,"W":52,"B":5,"QS":4}.get(freq,12)
        seas   = (1,1,0,s) if len(series)>=2*s else (0,0,0,0)
        res    = SARIMAX(series, order=(1,1,1), seasonal_order=seas,
                         enforce_stationarity=False, enforce_invertibility=False
                         ).fit(disp=False, maxiter=200)
        fc     = res.get_forecast(steps=horizon)
        ci     = fc.conf_int(alpha=0.10)
        dates  = pd.date_range(start=df["ds"].max()+_offset(freq), periods=horizon, freq=freq)
        fwd    = pd.DataFrame({"ds":dates,"yhat":fc.predicted_mean.values,
                               "lower":ci.iloc[:,0].values,"upper":ci.iloc[:,1].values})
        ins    = pd.DataFrame({"ds":series.index,"y_pred":res.fittedvalues.values})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception:
        return None, None

def run_xgboost(df, horizon, freq, n_lags=12):
    try:
        import xgboost as xgb
        s = df["y"].values.astype(float)
        def feat(arr, lags):
            X,y=[],[]
            for i in range(lags,len(arr)):
                w=arr[i-lags:i]
                X.append([*w,np.mean(w),np.std(w),np.min(w),np.max(w),w[-1]-w[0],w[-1]/(np.mean(w)+1e-9)])
                y.append(arr[i])
            return np.array(X),np.array(y)
        X,ya=feat(s,n_lags)
        sp=max(1,int(len(X)*0.15))
        mdl=xgb.XGBRegressor(n_estimators=500,learning_rate=0.04,max_depth=4,
                               subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
        mdl.fit(X[:-sp],ya[:-sp],eval_set=[(X[-sp:],ya[-sp:])],verbose=False)
        Xa,_=feat(s,n_lags)
        ins=pd.DataFrame({"ds":df["ds"].iloc[n_lags:].values,"y_pred":mdl.predict(Xa)})
        win=list(s[-n_lags:])
        yhats,lows,highs=[],[],[]
        ns=np.std(s)*0.04
        for step in range(horizon):
            w=np.array(win[-n_lags:])
            f_=np.array([*w,np.mean(w),np.std(w),np.min(w),np.max(w),w[-1]-w[0],w[-1]/(np.mean(w)+1e-9)]).reshape(1,-1)
            p=float(mdl.predict(f_)[0])
            sp2=ns*np.sqrt(step+1)*1.645
            yhats.append(p);lows.append(p-sp2);highs.append(p+sp2);win.append(p)
        dates=pd.date_range(start=df["ds"].max()+_offset(freq),periods=horizon,freq=freq)
        fwd=pd.DataFrame({"ds":dates,"yhat":yhats,"lower":lows,"upper":highs})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception:
        return None, None

def run_monte_carlo(df, horizon, freq, n_sims=1000, scenario="base"):
    s       = df["y"].values.astype(float)
    lr      = np.diff(np.log(s+1e-9))
    mu,sig  = np.mean(lr), np.std(lr)
    tm,ts   = {"best":(1.5,0.65),"base":(1.0,1.0),"worst":(0.3,1.5)}.get(scenario,(1.0,1.0))
    np.random.seed(99)
    last    = s[-1]
    paths   = np.zeros((n_sims,horizon))
    for i in range(n_sims):
        shocks=np.random.normal(mu*tm,sig*ts,horizon)
        paths[i]=last*np.exp(np.cumsum(shocks))
    dates = pd.date_range(start=df["ds"].max()+_offset(freq),periods=horizon,freq=freq)
    fwd   = pd.DataFrame({"ds":dates,
                           "yhat":np.percentile(paths,50,axis=0),
                           "lower":np.percentile(paths,10,axis=0),
                           "upper":np.percentile(paths,90,axis=0)})
    return fwd.reset_index(drop=True), paths

def ensemble(forecasts):
    valid = {k:v for k,v in forecasts.items() if v is not None}
    if len(valid)<2: return None
    base = list(valid.values())[0][["ds"]].copy()
    base["yhat"]  = np.mean([v["yhat"].values  for v in valid.values()],axis=0)
    base["lower"] = np.mean([v["lower"].values  for v in valid.values()],axis=0)
    base["upper"] = np.mean([v["upper"].values  for v in valid.values()],axis=0)
    return base

# ═══════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════

def title_style(text):
    return dict(text=text, font=dict(color="#9e9888",size=12,family="Syne"), x=0.01)

def fig_historical(df, mask, lo, hi):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
        line=dict(color=C["actual"],width=1.8),
        fill="tozeroy",fillcolor="rgba(107,107,130,0.05)"))
    if mask.any():
        adf=df[mask]
        fig.add_trace(go.Scatter(x=adf["ds"],y=adf["y"],mode="markers",name="Anomaly",
            marker=dict(color="#f25f5c",size=10,symbol="circle-open",line=dict(width=2,color="#f25f5c"))))
    fig.add_hline(y=df["y"].mean(),line_dash="dot",line_color="#252840",line_width=1,
                  annotation_text=f"μ={df['y'].mean():,.0f}",
                  annotation_font_color="#4a4a5e",annotation_font_size=10,
                  annotation_position="bottom right")
    fig.update_layout(**cly(title=title_style("Historical Series  ·  Anomaly Overlay"),height=340))
    return fig

def fig_yoy(df, freq):
    ppy={"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    yoy=df.set_index("ds")["y"].pct_change(ppy).dropna()*100
    colors=["#3ecf8e" if v>=0 else "#f25f5c" for v in yoy]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=yoy.index,y=yoy.values,marker_color=colors,marker_line=dict(width=0),name="YoY %"))
    fig.add_hline(y=0,line_color="#252840",line_width=1)
    fig.update_layout(**cly(title=title_style("Year-over-Year Growth (%)"),height=260,showlegend=False))
    return fig

def fig_returns(series):
    rets=series.pct_change().dropna()*100
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=rets,nbinsx=50,marker_color="rgba(200,169,81,0.7)",
                               marker_line=dict(width=0.5,color="#09090e"),name="Returns"))
    fig.add_vline(x=rets.mean(),line_dash="dash",line_color="#3ecf8e",line_width=1.5,
                  annotation_text=f"μ={rets.mean():.2f}%",
                  annotation_font_color="#3ecf8e",annotation_font_size=10,
                  annotation_position="top right")
    fig.add_vline(x=0,line_dash="solid",line_color="#252840",line_width=1)
    fig.update_layout(**cly(title=title_style("Return Distribution (%)"),height=290,showlegend=False))
    return fig

def fig_forecast(df, forecasts, ens):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
                             line=dict(color=C["actual"],width=2)))
    fig.add_vline(x=df["ds"].max(),line_dash="dot",line_color="#252840",line_width=1.5,
                  annotation_text="Forecast →",annotation_font_color="#4a4a5e",
                  annotation_font_size=10,annotation_position="top right")
    for name in ["Prophet","ARIMA","XGBoost","Monte Carlo"]:
        fwd=forecasts.get(name)
        if fwd is None: continue
        col=model_color(name)
        r,g,b=int(col[1:3],16),int(col[3:5],16),int(col[5:7],16)
        fig.add_trace(go.Scatter(
            x=pd.concat([fwd["ds"],fwd["ds"][::-1]]),
            y=pd.concat([fwd["upper"],fwd["lower"][::-1]]),
            fill="toself",showlegend=False,hoverinfo="skip",
            fillcolor=f"rgba({r},{g},{b},0.08)",line=dict(color="rgba(0,0,0,0)")))
        fig.add_trace(go.Scatter(x=fwd["ds"],y=fwd["yhat"],mode="lines",name=name,
            line=dict(color=col,width=2,
                      dash="dash" if name=="ARIMA" else "dot" if name=="Monte Carlo" else "solid")))
    if ens is not None:
        fig.add_trace(go.Scatter(x=ens["ds"],y=ens["yhat"],mode="lines",name="Ensemble",
                                 line=dict(color=C["ensemble"],width=2.5)))
    fig.update_layout(**cly(title=title_style("Multi-Model Forecast  ·  90% Confidence Intervals"),height=420))
    return fig

def fig_mc_fan(df, paths, dates, n_show=150):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Historical",
                             line=dict(color=C["actual"],width=2)))
    idx=np.random.choice(paths.shape[0],min(n_show,paths.shape[0]),replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(x=dates,y=paths[i],mode="lines",
            line=dict(color="rgba(242,95,92,0.04)",width=1),showlegend=False,hoverinfo="skip"))
    for p,lbl,col in [(90,"P90","#f25f5c"),(75,"P75","#f0a500"),
                       (50,"Median","#e8e2d5"),(25,"P25","#3ecf8e"),(10,"P10","#5b8dee")]:
        vals=np.percentile(paths,p,axis=0)
        fig.add_trace(go.Scatter(x=dates,y=vals,mode="lines",name=lbl,
            line=dict(color=col,width=2 if lbl=="Median" else 1.2,
                      dash="solid" if lbl=="Median" else "dot")))
    fig.add_vline(x=df["ds"].max(),line_dash="dot",line_color="#252840",line_width=1.5)
    fig.update_layout(**cly(title=title_style("Monte Carlo Fan  ·  Simulated Paths & Percentile Bands"),height=400))
    return fig

def fig_terminal(paths):
    t=paths[:,-1]
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=t,nbinsx=70,marker_color="rgba(242,95,92,0.55)",
                               marker_line=dict(width=0.3,color="#09090e"),name="Outcomes"))
    for p,col,lbl in [(5,"#f25f5c","P5 VaR"),(50,"#e8e2d5","Median"),(95,"#3ecf8e","P95")]:
        fig.add_vline(x=np.percentile(t,p),line_dash="dash",line_color=col,line_width=1.5,
                      annotation_text=lbl,annotation_font_color=col,annotation_font_size=10,
                      annotation_position="top")
    fig.update_layout(**cly(title=title_style("Terminal Value Distribution"),height=300,showlegend=False))
    return fig

def fig_accuracy(scores):
    models=[m for m in scores]
    mapes =[scores[m]["MAPE"]  for m in models]
    rmses =[scores[m]["RMSE"]  for m in models]
    cols  =[model_color(m) for m in models]
    fig   =make_subplots(rows=1,cols=2,subplot_titles=["MAPE % — Lower is Better","RMSE — Lower is Better"])
    fig.add_trace(go.Bar(x=models,y=mapes,marker_color=cols,showlegend=False,marker_line=dict(width=0)),row=1,col=1)
    fig.add_trace(go.Bar(x=models,y=rmses,marker_color=cols,showlegend=False,marker_line=dict(width=0)),row=1,col=2)
    lyt=cly(title=title_style("Model Accuracy  ·  In-Sample"),height=280)
    lyt["xaxis2"]=copy.deepcopy(lyt["xaxis"])
    lyt["yaxis2"]=copy.deepcopy(lyt["yaxis"])
    fig.update_layout(**lyt)
    for ann in fig.layout.annotations:
        ann.font.color="#4a4a5e"; ann.font.size=10; ann.font.family="Syne"
    return fig

def fig_fit(df, insample_dict):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
                             line=dict(color=C["actual"],width=2.5)))
    for name,ins in insample_dict.items():
        col=model_color(name)
        fig.add_trace(go.Scatter(x=ins["ds"],y=ins["y_pred"],mode="lines",name=name,
                                 line=dict(color=col,width=1.5,dash="dot")))
    fig.update_layout(**cly(title=title_style("In-Sample Fit vs Actual"),height=340))
    return fig

# ═══════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════

def sec(label):
    st.markdown(f'<div class="sec-div"><span class="sec-div-label">{label}</span><div class="sec-div-line"></div></div>',
                unsafe_allow_html=True)

def stat(k,v):
    return f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>'

def note(content, head="Analyst Note"):
    st.markdown(f'<div class="analyst-note"><div class="note-head">◈ {head}</div>{content}</div>',
                unsafe_allow_html=True)

def sig(kind, icon, text):
    st.markdown(f'<div class="signal-box {kind}"><span class="signal-icon">{icon}</span><span class="signal-text">{text}</span></div>',
                unsafe_allow_html=True)

def empty(msg, sub=""):
    st.markdown(f'<div class="empty-state"><div class="empty-state-icon">◈</div><div class="empty-state-msg">{msg}</div><div class="empty-state-sub">{sub}</div></div>',
                unsafe_allow_html=True)

def to_csv(df): return df.to_csv(index=False).encode()

def to_excel(sheets):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        for name,df in sheets.items(): df.to_excel(w,sheet_name=name[:31],index=False)
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <div class="sidebar-brand-title">Fin<em>Cast</em> Pro</div>
          <div class="sidebar-brand-sub">Institutional Forecasting Platform</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<span class="sb-section">Data Source</span>', unsafe_allow_html=True)
        mode = st.radio("Data Source",["Upload CSV","Use Sample Data"],label_visibility="collapsed")

        df = None
        if mode == "Upload CSV":
            f = st.file_uploader("CSV with columns: ds, y", type=["csv"],
                                 label_visibility="collapsed")
            if f:
                try:
                    df = pd.read_csv(f, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
                    st.markdown(f'<span class="badge badge-green">✓ {len(df):,} rows loaded</span>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Parse error: {e}")
        else:
            sample = st.selectbox("Dataset",["Monthly Revenue","Stock / Asset Price"],
                                  label_visibility="collapsed")
            df = gen_revenue() if "Revenue" in sample else gen_stock()
            st.markdown(f'<span class="badge badge-gold">Sample · {len(df):,} rows</span>',
                        unsafe_allow_html=True)

        if df is None: st.stop()

        st.markdown('<span class="sb-section">Forecast Parameters</span>', unsafe_allow_html=True)
        freq_map  = {"Monthly":"MS","Quarterly":"QS","Weekly":"W","Daily (Biz)":"B"}
        freq_lbl  = st.selectbox("Frequency", list(freq_map.keys()))
        freq      = freq_map[freq_lbl]
        max_h     = {"MS":24,"QS":8,"W":52,"B":252}[freq]
        def_h     = {"MS":12,"QS":4,"W":26,"B":63}[freq]
        horizon   = st.slider("Horizon (periods)", 1, max_h, def_h)

        st.markdown('<span class="sb-section">Models</span>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            use_p = st.checkbox("Prophet",  value=True)
            use_x = st.checkbox("XGBoost",  value=True)
        with c2:
            use_a = st.checkbox("ARIMA",    value=True)
            use_m = st.checkbox("Monte Carlo", value=True)

        n_sims = st.select_slider("MC Simulations",[500,1000,2000,5000],value=1000) if use_m else 1000

        st.markdown('<span class="sb-section">Scenario Planning</span>', unsafe_allow_html=True)
        scenario  = st.select_slider("Scenario",["Stress","Bear","Base","Bull","Upside"],value="Base")
        mc_scen   = {"Stress":"worst","Bear":"worst","Base":"base","Bull":"best","Upside":"best"}[scenario]
        scol      = {"Stress":"#f25f5c","Bear":"#f0a500","Base":"#c8a951","Bull":"#3ecf8e","Upside":"#5b8dee"}[scenario]
        st.markdown(f'<div style="margin-top:0.3rem;"><span class="badge" style="background:rgba(0,0,0,0.3);color:{scol};border:1px solid {scol}40;letter-spacing:0.1em;">{scenario.upper()}</span></div>',
                    unsafe_allow_html=True)

        st.markdown('<span class="sb-section">Anomaly Detection</span>', unsafe_allow_html=True)
        iqr_factor = st.slider("IQR Sensitivity", 1.0, 4.0, 2.0, 0.5)

        st.markdown("---")
        run = st.button("RUN FORECAST ENGINE", use_container_width=True)

        return df, freq, horizon, use_p, use_a, use_x, use_m, n_sims, mc_scen, scenario, iqr_factor, run

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    df, freq, horizon, use_p, use_a, use_x, use_m, n_sims, mc_scen, scenario, iqr_factor, run = sidebar()

    # Top Bar
    now = datetime.now()
    st.markdown(f"""
    <div class="fincast-topbar">
      <div style="display:flex;align-items:baseline;gap:0.75rem;">
        <span class="fincast-logo-mark">Fin<span>Cast</span> Pro</span>
        <span class="fincast-logo-tag">Institutional Forecasting</span>
      </div>
      <div class="fincast-meta">
        {now.strftime("%A, %d %B %Y")}&nbsp;&nbsp;|&nbsp;&nbsp;{now.strftime("%H:%M")} UTC<br>
        {len(df):,} obs &nbsp;·&nbsp; {freq} freq &nbsp;·&nbsp; {horizon}-period horizon &nbsp;·&nbsp; {scenario} scenario
      </div>
    </div>""", unsafe_allow_html=True)

    # Derived KPIs
    s    = df["y"]
    mask, lo, hi = detect_anomalies(s, iqr_factor)
    n_an = int(mask.sum())
    p_chg = (s.iloc[-1]-s.iloc[0])/s.iloc[0]*100
    r_chg = (s.iloc[-1]-s.iloc[-2])/s.iloc[-2]*100 if len(s)>1 else 0
    vol   = s.pct_change().std()*100
    _cagr = cagr(s,freq)
    _mdd  = max_drawdown(s)
    _sh   = sharpe(s,freq)

    rc = "pos" if r_chg>=0 else "neg"
    pc = "pos" if p_chg>=0 else "neg"
    dc = "neg" if _mdd<-10 else "neu"
    sc2= "pos" if _sh>=1  else ("warn" if _sh>=0 else "neg")
    ac = "warn" if n_an>0 else "pos"

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-cell accent">
        <div class="kpi-label">Latest Value</div>
        <div class="kpi-value">{s.iloc[-1]:,.0f}</div>
        <div class="kpi-sub {rc}">{"▲" if r_chg>=0 else "▼"} {abs(r_chg):.2f}% prior period</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Period Return</div>
        <div class="kpi-value {pc}">{p_chg:+.1f}%</div>
        <div class="kpi-sub neu">{len(s):,} observations</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">CAGR</div>
        <div class="kpi-value">{_cagr:.1f}%</div>
        <div class="kpi-sub neu">Annualised growth</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value {dc}">{_mdd:.1f}%</div>
        <div class="kpi-sub neu">Peak-to-trough</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value {sc2}">{_sh:.2f}</div>
        <div class="kpi-sub {ac}">{"⚠ " + str(n_an) + " anomaly" + ("" if n_an==1 else "ies") if n_an else "✓ Clean data"}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Tabs
    t1,t2,t3,t4,t5 = st.tabs([
        "  Data Intelligence  ",
        "  Forecast Engine  ",
        "  Risk & Monte Carlo  ",
        "  Model Validation  ",
        "  Report Export  ",
    ])

    # ══════════════════════════════════════════════════
    # TAB 1 — DATA INTELLIGENCE
    # ══════════════════════════════════════════════════
    with t1:
        sec("Historical Series")
        st.plotly_chart(fig_historical(df,mask,lo,hi), use_container_width=True)

        ca,cb = st.columns([3,2])
        with ca:
            sec("Year-over-Year Growth")
            st.plotly_chart(fig_yoy(df,freq), use_container_width=True)
        with cb:
            sec("Descriptive Statistics")
            html = "".join([
                stat("Observations",  f"{len(s):,}"),
                stat("Mean",          f"{s.mean():,.2f}"),
                stat("Median",        f"{s.median():,.2f}"),
                stat("Std Deviation", f"{s.std():,.2f}"),
                stat("Minimum",       f"{s.min():,.2f}"),
                stat("Maximum",       f"{s.max():,.2f}"),
                stat("Skewness",      f"{s.skew():.4f}"),
                stat("Kurtosis",      f"{s.kurt():.4f}"),
                stat("Volatility σ",  f"{vol:.2f}%"),
                stat("CAGR",          f"{_cagr:.2f}%"),
                stat("Max Drawdown",  f"{_mdd:.2f}%"),
                stat("Sharpe Ratio",  f"{_sh:.3f}"),
            ])
            st.markdown(f'<div class="panel">{html}</div>', unsafe_allow_html=True)

        cc,cd = st.columns([2,3])
        with cc:
            sec("Return Distribution")
            st.plotly_chart(fig_returns(s), use_container_width=True)
        with cd:
            sec(f"Anomaly Flags  ·  {n_an} detected")
            if n_an == 0:
                sig("bullish","✓","<strong>No anomalies detected.</strong> Data quality check passed at the current IQR sensitivity level.")
            else:
                adf = df[mask].copy()
                adf["pct_dev"] = (adf["y"]-s.mean())/s.mean()*100
                rows="".join([
                    f'<div class="anom-row"><span class="anom-date">{r["ds"].strftime("%Y-%m-%d")}</span>'
                    f'<span class="anom-val">{r["y"]:,.0f}</span>'
                    f'<span class="anom-dev">{r["pct_dev"]:+.1f}% vs μ</span></div>'
                    for _,r in adf.iterrows()
                ])
                st.markdown(f'<div style="max-height:230px;overflow-y:auto;">{rows}</div>',
                            unsafe_allow_html=True)

        skew_txt = ("positive skew — upside tail dominates"   if s.skew()>0.5
                    else "negative skew — downside tail dominates" if s.skew()<-0.5
                    else "near-symmetric distribution")
        note(
            f"Series shows <strong>{skew_txt}</strong>. CAGR of <strong>{_cagr:.1f}%</strong> "
            f"with annualised volatility <strong>{vol:.1f}%</strong>. Sharpe of "
            f"<strong>{_sh:.2f}</strong> "
            + ("— strong risk-adjusted return." if _sh>=1
               else "— adequate risk compensation." if _sh>=0
               else "— returns below the risk-free rate; review capital allocation.") +
            f" Max drawdown <strong>{_mdd:.1f}%</strong>"
            + (" — consider stop-loss or rebalancing triggers." if _mdd<-20 else ".") +
            (f" <strong>{n_an} anomalous observation{'s' if n_an!=1 else ''}</strong> detected — "
             "consider winsorising or excluding before modelling." if n_an
             else " No anomalies flagged — data is clean for modelling."),
            head="Data Intelligence Summary"
        )

    # ══════════════════════════════════════════════════
    # TAB 2 — FORECAST ENGINE
    # ══════════════════════════════════════════════════
    with t2:
        if not run:
            empty("Configure parameters and click  RUN FORECAST ENGINE",
                  "Select models · Set horizon · Choose scenario")
            return

        fcs, ins_d = {}, {}
        mc_paths = mc_fwd = None

        with st.spinner("Running models — this may take 15–30 seconds…"):
            if use_p:
                fwd,ins = run_prophet(df,horizon,freq)
                if fwd is not None: fcs["Prophet"]=fwd; ins_d["Prophet"]=ins
                else: st.warning("Prophet not installed: `pip install prophet`")
            if use_a:
                fwd,ins = run_arima(df,horizon,freq)
                if fwd is not None: fcs["ARIMA"]=fwd; ins_d["ARIMA"]=ins
                else: st.warning("statsmodels not installed.")
            if use_x:
                fwd,ins = run_xgboost(df,horizon,freq)
                if fwd is not None: fcs["XGBoost"]=fwd; ins_d["XGBoost"]=ins
                else: st.warning("XGBoost not installed.")
            if use_m:
                mc_fwd,mc_paths = run_monte_carlo(df,horizon,freq,n_sims,mc_scen)
                if mc_fwd is not None: fcs["Monte Carlo"]=mc_fwd

        if not fcs:
            st.error("No models ran. Install required packages.")
            return

        ens = ensemble(fcs)
        st.session_state.update({
            "fcs":fcs,"ins_d":ins_d,"mc_paths":mc_paths,"mc_fwd":mc_fwd,"ens":ens
        })

        sec("Multi-Model Forecast")
        st.plotly_chart(fig_forecast(df,fcs,ens), use_container_width=True)

        # Terminal KPI strip
        sec("Terminal Estimates")
        all_models = {**fcs, **({"Ensemble":ens} if ens is not None else {})}
        cols = st.columns(len(all_models))
        for i,(name,fwd) in enumerate(all_models.items()):
            t_val = fwd["yhat"].iloc[-1]
            t_chg = (t_val-s.iloc[-1])/s.iloc[-1]*100
            col   = model_color(name)
            cols[i].markdown(f"""
            <div class="panel" style="border-left:3px solid {col};padding:1rem 1.1rem;">
              <div class="kpi-label">{name}</div>
              <div class="kpi-value" style="font-size:1.2rem;">{t_val:,.0f}</div>
              <div class="kpi-sub {'pos' if t_chg>=0 else 'neg'}">{"▲" if t_chg>=0 else "▼"} {abs(t_chg):.1f}%</div>
              <div style="font-size:0.63rem;color:var(--cream-muted);margin-top:0.25rem;">{horizon}p · {scenario}</div>
            </div>""", unsafe_allow_html=True)

        # Forecast table
        sec("Forecast Table")
        rows=[]
        for d in fcs[list(fcs.keys())[0]]["ds"]:
            row={"Period":d.strftime("%Y-%m-%d")}
            for name,fwd in fcs.items():
                m=fwd[fwd["ds"]==d]
                if not m.empty:
                    row[name]=round(m["yhat"].values[0],2)
                    row[f"{name} Lo"]=round(m["lower"].values[0],2)
                    row[f"{name} Hi"]=round(m["upper"].values[0],2)
            if ens is not None:
                em=ens[ens["ds"]==d]
                if not em.empty: row["Ensemble"]=round(em["yhat"].values[0],2)
            rows.append(row)
        tbl=pd.DataFrame(rows)
        st.dataframe(tbl,hide_index=True,use_container_width=True)
        st.session_state["tbl"]=tbl
        st.session_state["df_h"]=df

        # Signals
        sec("Forecast Signals")
        if ens is not None:
            ec = (ens["yhat"].iloc[-1]-s.iloc[-1])/s.iloc[-1]*100
            if   ec>5:  sig("bullish","▲",f"<strong>Bullish signal:</strong> Consensus projects <strong>+{ec:.1f}%</strong> over {horizon} periods ({scenario} scenario).")
            elif ec<-5: sig("bearish","▼",f"<strong>Bearish signal:</strong> Consensus projects <strong>{ec:.1f}%</strong> over {horizon} periods.")
            else:       sig("neutral","◆",f"<strong>Neutral:</strong> Ensemble projects modest <strong>{ec:+.1f}%</strong> movement over {horizon} periods.")
        if n_an>0:
            sig("caution","⚠",f"<strong>Data Quality Alert:</strong> {n_an} anomalous observations in training data may widen confidence intervals.")

    # ══════════════════════════════════════════════════
    # TAB 3 — RISK & MONTE CARLO
    # ══════════════════════════════════════════════════
    with t3:
        if not run:
            empty("Run the forecast engine first.")
            return
        mc_paths = st.session_state.get("mc_paths")
        mc_fwd   = st.session_state.get("mc_fwd")
        if not use_m or mc_paths is None:
            st.info("Enable Monte Carlo in the sidebar to unlock this view.")
            return

        dates = mc_fwd["ds"]
        sec(f"Monte Carlo Fan  ·  {n_sims:,} Simulations  ·  {scenario} Scenario")
        st.plotly_chart(fig_mc_fan(df,mc_paths,dates), use_container_width=True)

        ca,cb = st.columns([3,2])
        with ca:
            sec("Terminal Value Distribution")
            st.plotly_chart(fig_terminal(mc_paths), use_container_width=True)
        with cb:
            tv = mc_paths[:,-1]
            var95  = np.percentile(tv,5)
            cvar95 = tv[tv<=var95].mean()
            p_loss = np.mean(tv<s.iloc[-1])*100
            p_10up = np.mean(tv>s.iloc[-1]*1.1)*100

            sec("Risk Metrics")
            rh = "".join([
                stat("VaR (95%)",       f"{var95:,.0f}"),
                stat("CVaR / ES (95%)", f"{cvar95:,.0f}"),
                stat("P(loss)",         f"{p_loss:.1f}%"),
                stat("P(gain >10%)",    f"{p_10up:.1f}%"),
                stat("Median",          f"{np.percentile(tv,50):,.0f}"),
                stat("P10",             f"{np.percentile(tv,10):,.0f}"),
                stat("P90",             f"{np.percentile(tv,90):,.0f}"),
                stat("Simulations",     f"{n_sims:,}"),
                stat("Scenario",        scenario),
            ])
            st.markdown(f'<div class="panel">{rh}</div>', unsafe_allow_html=True)

            sec("Percentile Table")
            pdf = pd.DataFrame({
                "Pctl": [f"P{p}" for p in [1,5,10,25,50,75,90,95,99]],
                "Value": [f"{np.percentile(tv,p):,.0f}" for p in [1,5,10,25,50,75,90,95,99]],
                "vs Last": [f"{(np.percentile(tv,p)/s.iloc[-1]-1)*100:+.1f}%" for p in [1,5,10,25,50,75,90,95,99]],
            })
            st.dataframe(pdf, hide_index=True, use_container_width=True)

        note(
            f"<strong>{n_sims:,} GBM simulations</strong> under <strong>{scenario}</strong> scenario. "
            f"VaR (95%): <strong>{var95:,.0f}</strong> · CVaR: <strong>{cvar95:,.0f}</strong>. "
            f"Probability of decline: <strong>{p_loss:.1f}%</strong>. "
            f"Probability of +10% gain: <strong>{p_10up:.1f}%</strong>. "
            + ("Recommend stress-testing under alternative macro assumptions." if scenario=="Base" else
               f"The {scenario} scenario applies adjusted drift and volatility parameters."),
            head="Risk & Simulation Summary"
        )

    # ══════════════════════════════════════════════════
    # TAB 4 — MODEL VALIDATION
    # ══════════════════════════════════════════════════
    with t4:
        if not run:
            empty("Run the forecast engine first.")
            return
        ins_d = st.session_state.get("ins_d",{})
        if not ins_d:
            st.info("Select at least one parametric model (Prophet / ARIMA / XGBoost).")
            return

        scores={}
        for name,ins in ins_d.items():
            merged=df.merge(ins,on="ds",how="inner")
            if merged.empty: continue
            a,f_=merged["y"].values,merged["y_pred"].values
            scores[name]={"MAPE":round(mape(a,f_),3),"sMAPE":round(smape(a,f_),3),"RMSE":round(rmse(a,f_),2)}
        st.session_state["scores"]=scores

        if not scores:
            st.warning("Could not compute scores.")
            return

        ranked=sorted(scores,key=lambda m:scores[m]["MAPE"])
        icons=["gold","silver","bronze"]

        sec("Model Leaderboard")
        rows_html=""
        for i,name in enumerate(ranked):
            sc=scores[name]
            col=model_color(name)
            conf=max(0,min(100,100-sc["MAPE"]*5))
            best_tag='<span class="badge badge-gold" style="margin-left:0.5rem;">BEST FIT</span>' if i==0 else ""
            rows_html+=f"""<tr {"class='lb-winner'" if i==0 else ""}>
              <td><span class="lb-rank {icons[i] if i<3 else ''}  ">{i+1}</span>
                  <span class="model-dot" style="background:{col};"></span>{name}{best_tag}</td>
              <td>{sc['MAPE']:.3f}%</td><td>{sc['sMAPE']:.3f}%</td><td>{sc['RMSE']:,.2f}</td>
              <td><div style="display:flex;align-items:center;gap:6px;">
                <div style="width:{conf:.0f}px;height:4px;background:{col};border-radius:2px;max-width:80px;"></div>
                <span style="font-size:0.7rem;color:var(--cream-muted);">{conf:.0f}/100</span>
              </div></td></tr>"""
        st.markdown(f"""
        <div class="panel">
          <table class="lb-table">
            <thead><tr><th>Model</th><th>MAPE (%)</th><th>sMAPE (%)</th><th>RMSE</th><th>Confidence</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        sec("Accuracy Comparison")
        st.plotly_chart(fig_accuracy(scores), use_container_width=True)

        sec("In-Sample Fit vs Actual")
        st.plotly_chart(fig_fit(df,ins_d), use_container_width=True)

        best=ranked[0]
        gap_txt=""
        if len(ranked)>1:
            gap=abs(scores[ranked[0]]["MAPE"]-scores[ranked[1]]["MAPE"])
            gap_txt=f" Gap to runner-up: <strong>{gap:.2f}pp</strong> — " + \
                    ("models closely matched; ensemble recommended." if gap<1 else "prefer the top model.")
        note(
            f"<strong>{best}</strong> achieves the lowest MAPE of <strong>{scores[best]['MAPE']:.2f}%</strong> "
            f"and RMSE of <strong>{scores[best]['RMSE']:,.2f}</strong> on in-sample data.{gap_txt} "
            "Recommended next steps: <strong>(1)</strong> Walk-forward validation on the last 20% of data. "
            "<strong>(2)</strong> Ensemble the top two models to reduce out-of-sample variance. "
            "<strong>(3)</strong> Retrain on a rolling window to capture regime shifts.",
            head="Validation & Deployment Guidance"
        )

    # ══════════════════════════════════════════════════
    # TAB 5 — REPORT EXPORT
    # ══════════════════════════════════════════════════
    with t5:
        sec("Download Outputs")
        ds = datetime.today().strftime("%Y%m%d")
        tbl = st.session_state.get("tbl")
        hist_exp = df.copy()
        hist_exp["anomaly_flag"] = mask.astype(int)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown('<div class="export-card">', unsafe_allow_html=True)
            st.markdown('<div class="export-card-title">📄 Forecast Table — CSV</div>', unsafe_allow_html=True)
            st.markdown('<div class="export-card-desc">All models with confidence intervals. Compatible with Excel, Tableau, PowerBI, and Python.</div>', unsafe_allow_html=True)
            if tbl is not None:
                st.download_button("↓ Download CSV", to_csv(tbl),
                                   f"FinCast_Forecast_{ds}.csv","text/csv")
            else:
                st.caption("Run forecast first.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="export-card">', unsafe_allow_html=True)
            st.markdown('<div class="export-card-title">📊 Historical Data — CSV</div>', unsafe_allow_html=True)
            st.markdown('<div class="export-card-desc">Cleaned historical series with anomaly flags. Ready for audit trail and data lineage.</div>', unsafe_allow_html=True)
            st.download_button("↓ Download CSV", to_csv(hist_exp),
                               f"FinCast_Historical_{ds}.csv","text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="export-card">', unsafe_allow_html=True)
            st.markdown('<div class="export-card-title">📁 Full Report — Excel</div>', unsafe_allow_html=True)
            st.markdown('<div class="export-card-desc">Multi-sheet workbook: Historical · Forecast · Model Scores · MC Summary. Board-deck ready.</div>', unsafe_allow_html=True)
            if tbl is not None:
                sheets={"Historical":hist_exp,"Forecast":tbl}
                sc=st.session_state.get("scores",{})
                if sc:
                    sdf=pd.DataFrame(sc).T.reset_index()
                    sdf.columns=["Model","MAPE","sMAPE","RMSE"]
                    sheets["Model Accuracy"]=sdf
                mc_fwd2=st.session_state.get("mc_fwd")
                if mc_fwd2 is not None: sheets["MC Summary"]=mc_fwd2[["ds","yhat","lower","upper"]]
                st.download_button("↓ Download Excel",to_excel(sheets),
                                   f"FinCast_Report_{ds}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.caption("Run forecast first.")
            st.markdown('</div>', unsafe_allow_html=True)

        if tbl is not None:
            sec("Report Summary")
            sc=st.session_state.get("scores",{})
            best_m=min(sc,key=lambda m:sc[m]["MAPE"]) if sc else "N/A"
            ens2=st.session_state.get("ens")
            ens_t=f"{ens2['yhat'].iloc[-1]:,.0f}" if ens2 is not None else "N/A"
            note(
                f"Report generated <strong>{now.strftime('%d %B %Y, %H:%M UTC')}</strong>. "
                f"Dataset: <strong>{len(df):,} observations</strong> · {freq} frequency. "
                f"Horizon: <strong>{horizon} periods</strong> · <strong>{scenario}</strong> scenario. "
                f"Best-fit model: <strong>{best_m}</strong>"
                + (f" (MAPE {sc[best_m]['MAPE']:.2f}%)" if best_m!="N/A" else "") + ". "
                f"Ensemble terminal: <strong>{ens_t}</strong>. "
                f"CAGR <strong>{_cagr:.1f}%</strong> · MDD <strong>{_mdd:.1f}%</strong> · Sharpe <strong>{_sh:.2f}</strong>. "
                + (f"<strong>{n_an} anomalies</strong> flagged." if n_an else "No anomalies detected."),
                head="Automated Report Summary"
            )

if __name__=="__main__":
    main()
