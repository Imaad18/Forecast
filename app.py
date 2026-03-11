"""
FinCast Pro v2 — Institutional Financial Forecasting Platform
New in v2:
  • Yahoo Finance / ticker data source
  • Auto frequency detection from uploaded CSV
  • Outlier winsorisation toggle (pre-fit treatment)
  • Per-model live progress status
  • Walk-forward backtesting (rolling OOS validation)
  • Residual diagnostics — ACF plot + bias/variance decomposition
  • Regime / structural-break detection overlay
  • Confidence interval toggle per model
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, io, copy
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FinCast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ───────────────────────────────────────────────────────────
G   = "#c8a951"
CR  = "#e8e2d5"
CDM = "#9e9888"
CMT = "#4a4a5e"
BG0 = "#09090e"
BG1 = "#0e0f17"
BG2 = "#13141e"
BD  = "#1e2030"
BD2 = "#252840"
POS = "#3ecf8e"
NEG = "#f25f5c"
AMB = "#f0a500"
BLU = "#5b8dee"
PRP = "#d4a0ff"
MCLR = {"Prophet":G,"ARIMA":BLU,"XGBoost":POS,"Monte Carlo":NEG,"Ensemble":PRP}
def mc(n): return MCLR.get(n, CDM)

# ── Chart base ───────────────────────────────────────────────────────────────
_BL = dict(
    paper_bgcolor=BG0, plot_bgcolor=BG0,
    font=dict(family="monospace", color=CMT, size=11),
    xaxis=dict(gridcolor=BG2, linecolor=BD, zeroline=False,
               tickfont=dict(color=CMT, size=10),
               showspikes=True, spikecolor=BD2, spikethickness=1),
    yaxis=dict(gridcolor=BG2, linecolor=BD, zeroline=False,
               tickfont=dict(color=CMT, size=10)),
    legend=dict(bgcolor="rgba(14,15,23,0.9)", bordercolor=BD2, borderwidth=1,
                font=dict(color=CDM, size=10)),
    margin=dict(l=10, r=10, t=38, b=10),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=BG2, bordercolor=BD2,
                    font=dict(color=CR, size=11), namelength=-1),
)
def cl(**kw):
    d=copy.deepcopy(_BL); d.update(kw); return d
def ttl(t):
    return dict(text=t, font=dict(color=CDM, size=12), x=0.01)

# ── Data helpers ─────────────────────────────────────────────────────────────
def gen_revenue(n=48):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today().replace(day=1), periods=n, freq="MS")
    trend = np.linspace(480_000, 880_000, n)
    season = 75_000*np.sin(2*np.pi*np.arange(n)/12 - np.pi/3)
    vals = np.clip(trend+season+np.random.normal(0,22_000,n), 0, None)
    vals[n-4] *= 0.14
    return pd.DataFrame({"ds":dates,"y":vals.round(0)})

def gen_stock(n=500):
    np.random.seed(7)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    price = 120*np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    return pd.DataFrame({"ds":dates,"y":price.round(4)})

@st.cache_data(show_spinner=False)
def fetch_yahoo(ticker, period="5y"):
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty: return None, "No data returned for that ticker."
        raw = raw.reset_index()
        # Handle MultiIndex columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] if c[1]=='' else c[0] for c in raw.columns]
        df = raw[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        df["y"] = df["y"].astype(float)
        return df.sort_values("ds").reset_index(drop=True), None
    except ImportError:
        return None, "yfinance not installed — add it to requirements.txt"
    except Exception as e:
        return None, str(e)

def auto_detect_freq(df):
    """Infer frequency from median gap between dates."""
    if len(df) < 3: return "MS", "Monthly"
    gaps = df["ds"].sort_values().diff().dropna().dt.days
    med = gaps.median()
    if med <= 2:    return "B",  "Daily (Biz)"
    if med <= 8:    return "W",  "Weekly"
    if med <= 35:   return "MS", "Monthly"
    return "QS", "Quarterly"

def winsorise(df, factor=2.0):
    """Replace IQR outliers with boundary values before fitting."""
    s = df["y"].copy()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3-q1
    lo, hi = q1-factor*iqr, q3+factor*iqr
    s = s.clip(lo, hi)
    return df.assign(y=s)

def detect_anomalies(s, factor=2.0):
    q1,q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3-q1
    return (s < q1-factor*iqr)|(s > q3+factor*iqr), q1-factor*iqr, q3+factor*iqr

def detect_regimes(s, min_gap=6):
    """Simple CUSUM-style structural break detection."""
    z = (s - s.mean()) / (s.std()+1e-9)
    cusum = z.cumsum()
    breaks = []
    last = 0
    for i in range(min_gap, len(cusum)-min_gap):
        if abs(cusum.iloc[i] - cusum.iloc[last]) > 2.5:
            breaks.append(i)
            last = i
    return breaks

def cagr(s, freq):
    ppy = {"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    n = len(s)/ppy
    if n<=0 or s.iloc[0]<=0: return 0.0
    return ((s.iloc[-1]/s.iloc[0])**(1/n)-1)*100

def max_dd(s):
    return ((s-s.cummax())/s.cummax()).min()*100

def sharpe(s, freq, rfr=0.05):
    ppy = {"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    r = s.pct_change().dropna()
    if r.std()==0: return 0.0
    return (r.mean()*ppy - rfr)/(r.std()*np.sqrt(ppy))

def mape(a,f):
    a,f=np.array(a),np.array(f); m=(a!=0)&~np.isnan(a)&~np.isnan(f)
    return 100*np.mean(np.abs((a[m]-f[m])/a[m]))

def smape(a,f):
    a,f=np.array(a),np.array(f); m=~(np.isnan(a)|np.isnan(f))
    return 100*np.mean(2*np.abs(f[m]-a[m])/(np.abs(a[m])+np.abs(f[m])+1e-9))

def rmse(a,f):
    a,f=np.array(a),np.array(f); m=~(np.isnan(a)|np.isnan(f))
    return np.sqrt(np.mean((a[m]-f[m])**2))

def bias_variance(a, f):
    """Return bias (mean error), variance component, noise component."""
    e = np.array(f)-np.array(a)
    bias = np.mean(e)
    var  = np.var(f)
    noise= np.var(a)
    return float(bias), float(var), float(noise)

# ── Models ───────────────────────────────────────────────────────────────────
def _off(freq):
    try: return pd.tseries.frequencies.to_offset(freq)
    except: return pd.DateOffset(months=1)

@st.cache_data(show_spinner=False)
def run_prophet(df_hash, df, horizon, freq):
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        m = Prophet(yearly_seasonality=6, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.90,
                    changepoint_prior_scale=0.05, n_changepoints=15,
                    uncertainty_samples=200)
        m.fit(df[["ds","y"]])
        fut = m.make_future_dataframe(periods=horizon, freq=freq)
        fc  = m.predict(fut)
        fwd = fc[fc["ds"]>df["ds"].max()][["ds","yhat","yhat_lower","yhat_upper"]].copy()
        fwd.columns = ["ds","yhat","lower","upper"]
        ins = fc[fc["ds"]<=df["ds"].max()][["ds","yhat"]].rename(columns={"yhat":"y_pred"})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception: return None, None

@st.cache_data(show_spinner=False)
def run_arima(df_hash, df, horizon, freq):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        s  = df.set_index("ds")["y"]
        sp = {"MS":12,"W":52,"B":5,"QS":4}.get(freq,12)
        seas = (1,1,0,sp) if len(s)>=3*sp else (0,0,0,0)
        res  = SARIMAX(s, order=(1,1,1), seasonal_order=seas,
                       enforce_stationarity=False, enforce_invertibility=False
                       ).fit(disp=False, maxiter=75, method="lbfgs")
        fc  = res.get_forecast(steps=horizon)
        ci  = fc.conf_int(alpha=0.10)
        dates = pd.date_range(start=df["ds"].max()+_off(freq), periods=horizon, freq=freq)
        fwd = pd.DataFrame({"ds":dates,"yhat":fc.predicted_mean.values,
                            "lower":ci.iloc[:,0].values,"upper":ci.iloc[:,1].values})
        ins = pd.DataFrame({"ds":s.index,"y_pred":res.fittedvalues.values})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception: return None, None

@st.cache_data(show_spinner=False)
def run_xgboost(df_hash, df, horizon, freq, n_lags=12):
    try:
        import xgboost as xgb
        s = df["y"].values.astype(float)
        def feat(arr, lags):
            X,y=[],[]
            for i in range(lags,len(arr)):
                w=arr[i-lags:i]
                X.append([*w,np.mean(w),np.std(w),np.min(w),np.max(w),
                           w[-1]-w[0],w[-1]/(np.mean(w)+1e-9)])
                y.append(arr[i])
            return np.array(X),np.array(y)
        X,ya = feat(s,n_lags); sp=max(1,int(len(X)*0.15))
        mdl = xgb.XGBRegressor(n_estimators=150, learning_rate=0.08, max_depth=3,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                verbosity=0, tree_method="hist")
        mdl.fit(X[:-sp], ya[:-sp], verbose=False)
        Xa,_ = feat(s,n_lags)
        ins = pd.DataFrame({"ds":df["ds"].iloc[n_lags:].values,"y_pred":mdl.predict(Xa)})
        win=list(s[-n_lags:]); yhats,lows,highs=[],[],[]; ns=np.std(s)*0.04
        for step in range(horizon):
            w=np.array(win[-n_lags:])
            f_=np.array([*w,np.mean(w),np.std(w),np.min(w),np.max(w),
                         w[-1]-w[0],w[-1]/(np.mean(w)+1e-9)]).reshape(1,-1)
            p=float(mdl.predict(f_)[0]); sp2=ns*np.sqrt(step+1)*1.645
            yhats.append(p); lows.append(p-sp2); highs.append(p+sp2); win.append(p)
        dates=pd.date_range(start=df["ds"].max()+_off(freq),periods=horizon,freq=freq)
        fwd=pd.DataFrame({"ds":dates,"yhat":yhats,"lower":lows,"upper":highs})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception: return None, None

@st.cache_data(show_spinner=False)
def run_monte_carlo(df_hash, df, horizon, freq, n_sims=1000, scenario="base"):
    s=df["y"].values.astype(float)
    lr=np.diff(np.log(s+1e-9)); mu,sig=np.mean(lr),np.std(lr)
    tm,ts={"best":(1.5,0.65),"base":(1.0,1.0),"worst":(0.3,1.5)}.get(scenario,(1.0,1.0))
    np.random.seed(99); last=s[-1]
    shocks=np.random.normal(mu*tm, sig*ts, size=(n_sims,horizon))
    paths=last*np.exp(np.cumsum(shocks,axis=1))
    dates=pd.date_range(start=df["ds"].max()+_off(freq),periods=horizon,freq=freq)
    fwd=pd.DataFrame({"ds":dates,"yhat":np.percentile(paths,50,axis=0),
                      "lower":np.percentile(paths,10,axis=0),"upper":np.percentile(paths,90,axis=0)})
    return fwd.reset_index(drop=True), paths

@st.cache_data(show_spinner=False)
def run_walk_forward(df_hash, df, freq, n_splits=5, use_p=True, use_a=True, use_x=True):
    """Rolling-origin walk-forward validation."""
    results = []
    n = len(df)
    min_train = max(24, n//3)
    step = max(1, (n-min_train)//n_splits)
    splits = [(min_train + i*step, min_train + i*step + step)
              for i in range(n_splits) if min_train + i*step + step <= n]
    if not splits: return pd.DataFrame()

    for train_end, test_end in splits:
        train = df.iloc[:train_end].copy()
        test  = df.iloc[train_end:test_end].copy()
        h     = len(test)
        dh    = int(str(train["ds"].max().date()).replace("-","")) * 1000 + h
        fold = {"fold": f"{train['ds'].max().strftime('%b %Y')}",
                "train_n": train_end, "test_n": h}
        for name, fn, flag in [("Prophet",run_prophet,use_p),
                                ("ARIMA",  run_arima,  use_a),
                                ("XGBoost",run_xgboost,use_x)]:
            if not flag: continue
            try:
                fwd, _ = fn(dh, train, h, freq)
                if fwd is None: continue
                preds = fwd["yhat"].values[:h]
                actual= test["y"].values[:h]
                fold[f"{name}_MAPE"] = round(mape(actual,preds),2)
                fold[f"{name}_RMSE"] = round(rmse(actual,preds),2)
            except: pass
        results.append(fold)
    return pd.DataFrame(results)

def build_ensemble(forecasts):
    valid={k:v for k,v in forecasts.items() if v is not None}
    if len(valid)<2: return None
    base=list(valid.values())[0][["ds"]].copy()
    base["yhat"] =np.mean([v["yhat"].values for v in valid.values()],axis=0)
    base["lower"]=np.mean([v["lower"].values for v in valid.values()],axis=0)
    base["upper"]=np.mean([v["upper"].values for v in valid.values()],axis=0)
    return base

# ── Charts ───────────────────────────────────────────────────────────────────
def fig_hist(df, mask, regimes, show_regimes):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
        line=dict(color=CDM,width=1.8),fill="tozeroy",fillcolor="rgba(107,107,130,0.05)"))
    if mask.any():
        adf=df[mask]
        fig.add_trace(go.Scatter(x=adf["ds"],y=adf["y"],mode="markers",name="Anomaly",
            marker=dict(color=NEG,size=10,symbol="circle-open",line=dict(width=2,color=NEG))))
    if show_regimes:
        for bi in regimes:
            if bi < len(df):
                bx = df["ds"].iloc[bi]
                fig.add_vline(x=bx, line_dash="dot", line_color=AMB, line_width=1.2)
                fig.add_annotation(x=bx, y=1, yref="paper", text="Regime ↕",
                                   showarrow=False, font=dict(color=AMB,size=9),
                                   xanchor="left", yanchor="top", xshift=3)
    mean_val=df["y"].mean()
    fig.add_hline(y=mean_val, line_dash="dot", line_color=BD2, line_width=1)
    fig.add_annotation(x=1, y=mean_val, xref="paper", text=f"μ {mean_val:,.0f}",
                       showarrow=False, font=dict(color=CMT,size=10),
                       xanchor="right", yanchor="bottom")
    fig.update_layout(**cl(title=ttl("Historical Series · Anomaly & Regime Detection"),height=360))
    return fig

def fig_yoy(df, freq):
    ppy={"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    yoy=df.set_index("ds")["y"].pct_change(ppy).dropna()*100
    colors=[POS if v>=0 else NEG for v in yoy]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=yoy.index,y=yoy.values,marker_color=colors,marker_line_width=0,name="YoY%"))
    fig.add_hline(y=0, line_color=BD2, line_width=1)
    fig.update_layout(**cl(title=ttl("Year-over-Year Growth (%)"),height=260,showlegend=False))
    return fig

def fig_returns(s):
    r=s.pct_change().dropna()*100
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=r,nbinsx=50,marker_color="rgba(200,169,81,0.65)",
                               marker_line_width=0,name="Returns"))
    mr=float(r.mean())
    fig.add_vline(x=mr, line_dash="dash", line_color=POS, line_width=1.5)
    fig.add_annotation(x=mr,y=1,yref="paper",text=f"μ={mr:.2f}%",
                       showarrow=False,font=dict(color=POS,size=10),
                       xanchor="left",yanchor="top",xshift=4)
    fig.add_vline(x=0.0, line_dash="solid", line_color=BD2, line_width=1)
    fig.update_layout(**cl(title=ttl("Return Distribution (%)"),height=290,showlegend=False))
    return fig

def fig_forecast(df, forecasts, ens, show_ci):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
                             line=dict(color=CDM,width=2)))
    fig.add_vline(x=df["ds"].max(), line_dash="dot", line_color=BD2, line_width=1.5)
    for name in ["Prophet","ARIMA","XGBoost","Monte Carlo"]:
        fwd=forecasts.get(name)
        if fwd is None: continue
        col=mc(name); r,g,b=int(col[1:3],16),int(col[3:5],16),int(col[5:7],16)
        if show_ci:
            fig.add_trace(go.Scatter(
                x=pd.concat([fwd["ds"],fwd["ds"][::-1]]),
                y=pd.concat([fwd["upper"],fwd["lower"][::-1]]),
                fill="toself",showlegend=False,hoverinfo="skip",
                fillcolor=f"rgba({r},{g},{b},0.07)",line=dict(color="rgba(0,0,0,0)")))
        fig.add_trace(go.Scatter(x=fwd["ds"],y=fwd["yhat"],mode="lines",name=name,
            line=dict(color=col,width=2,
                      dash="dash" if name=="ARIMA" else "dot" if name=="Monte Carlo" else "solid")))
    if ens is not None:
        fig.add_trace(go.Scatter(x=ens["ds"],y=ens["yhat"],mode="lines",name="Ensemble",
                                 line=dict(color=PRP,width=2.5)))
    ci_txt="90% CI" if show_ci else "No CI"
    fig.update_layout(**cl(title=ttl(f"Multi-Model Forecast · {ci_txt}"),height=420))
    return fig

def fig_mc_fan(df, paths, dates):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Historical",
                             line=dict(color=CDM,width=2)))
    idx=np.random.choice(paths.shape[0],min(100,paths.shape[0]),replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(x=dates,y=paths[i],mode="lines",
            line=dict(color="rgba(242,95,92,0.04)",width=1),showlegend=False,hoverinfo="skip"))
    for p,lbl,col in [(90,"P90",NEG),(75,"P75",AMB),(50,"Median",CR),(25,"P25",POS),(10,"P10",BLU)]:
        fig.add_trace(go.Scatter(x=dates,y=np.percentile(paths,p,axis=0),mode="lines",name=lbl,
            line=dict(color=col,width=2 if lbl=="Median" else 1.2,
                      dash="solid" if lbl=="Median" else "dot")))
    fig.add_vline(x=df["ds"].max(), line_dash="dot", line_color=BD2, line_width=1.5)
    fig.update_layout(**cl(title=ttl("Monte Carlo Fan · Simulated Paths"),height=400))
    return fig

def fig_terminal(paths):
    t=paths[:,-1]
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=t,nbinsx=70,marker_color="rgba(242,95,92,0.55)",
                               marker_line_width=0))
    for p,col,lbl in [(5,NEG,"P5"),(50,CR,"Median"),(95,POS,"P95")]:
        pval=float(np.percentile(t,p))
        fig.add_vline(x=pval, line_dash="dash", line_color=col, line_width=1.5)
        fig.add_annotation(x=pval,y=1,yref="paper",text=lbl,
                           showarrow=False,font=dict(color=col,size=10),
                           xanchor="left",yanchor="top",xshift=4)
    fig.update_layout(**cl(title=ttl("Terminal Value Distribution"),height=300,showlegend=False))
    return fig

def fig_accuracy(scores):
    models=list(scores)
    mapes=[scores[m]["MAPE"] for m in models]; rmses=[scores[m]["RMSE"] for m in models]
    cols=[mc(m) for m in models]
    fig=make_subplots(rows=1,cols=2,subplot_titles=["MAPE % (lower=better)","RMSE (lower=better)"])
    fig.add_trace(go.Bar(x=models,y=mapes,marker_color=cols,showlegend=False,marker_line_width=0),row=1,col=1)
    fig.add_trace(go.Bar(x=models,y=rmses,marker_color=cols,showlegend=False,marker_line_width=0),row=1,col=2)
    lyt=cl(title=ttl("Model Accuracy · In-Sample"),height=280)
    lyt["xaxis2"]=copy.deepcopy(lyt["xaxis"]); lyt["yaxis2"]=copy.deepcopy(lyt["yaxis"])
    fig.update_layout(**lyt)
    for ann in fig.layout.annotations: ann.font.color=CMT; ann.font.size=10
    return fig

def fig_fit(df, ins_d):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
                             line=dict(color=CDM,width=2.5)))
    for name,ins in ins_d.items():
        fig.add_trace(go.Scatter(x=ins["ds"],y=ins["y_pred"],mode="lines",name=name,
                                 line=dict(color=mc(name),width=1.5,dash="dot")))
    fig.update_layout(**cl(title=ttl("In-Sample Fit vs Actual"),height=340))
    return fig

def fig_residuals(df, ins_d):
    """Residuals over time + ACF for each model."""
    if not ins_d: return None
    n_models = len(ins_d)
    fig = make_subplots(rows=n_models, cols=2,
                        subplot_titles=[f"{n} Residuals" if i%2==0 else f"{n} ACF"
                                        for n in ins_d for i in range(2)],
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    for row, (name, ins) in enumerate(ins_d.items(), 1):
        merged = df.merge(ins, on="ds", how="inner")
        if merged.empty: continue
        resid = (merged["y"] - merged["y_pred"]).values
        col_c = mc(name)
        # Residuals over time
        fig.add_trace(go.Scatter(x=merged["ds"], y=resid, mode="lines",
                                 line=dict(color=col_c, width=1.2), name=name, showlegend=False),
                      row=row, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color=BD2, line_width=1, row=row, col=1)
        # ACF
        max_lag = min(20, len(resid)//3)
        acf_vals = [1.0] + [float(pd.Series(resid).autocorr(lag=l)) for l in range(1, max_lag+1)]
        ci_bound = 1.96 / np.sqrt(len(resid))
        lags = list(range(len(acf_vals)))
        bar_colors = [col_c if abs(v)>ci_bound else CMT for v in acf_vals]
        fig.add_trace(go.Bar(x=lags, y=acf_vals, marker_color=bar_colors,
                             marker_line_width=0, name=f"{name} ACF", showlegend=False),
                      row=row, col=2)
        # CI bands
        for y_ci in [ci_bound, -ci_bound]:
            fig.add_hline(y=y_ci, line_dash="dot", line_color=BD2, line_width=1, row=row, col=2)

    h = max(280, 240*n_models)
    lyt = cl(title=ttl("Residual Diagnostics · Time Series & ACF"), height=h, showlegend=False)
    # Propagate axis styles to all subplots
    for i in range(1, n_models*2+1):
        k = "" if i==1 else str(i)
        lyt[f"xaxis{k}"] = copy.deepcopy(_BL["xaxis"])
        lyt[f"yaxis{k}"] = copy.deepcopy(_BL["yaxis"])
    fig.update_layout(**lyt)
    for ann in fig.layout.annotations: ann.font.color=CMT; ann.font.size=10
    return fig

def fig_bias_variance(scores, ins_d, df):
    """Decompose MSE into bias² + variance components."""
    names, bias2s, vars_, noises = [], [], [], []
    noise = float(np.var(df["y"].values))
    for name, ins in ins_d.items():
        merged = df.merge(ins, on="ds", how="inner")
        if merged.empty: continue
        b, v, _ = bias_variance(merged["y"].values, merged["y_pred"].values)
        names.append(name); bias2s.append(b**2); vars_.append(v); noises.append(noise)
    if not names: return None
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Bias²",     x=names, y=bias2s,  marker_color=NEG, marker_line_width=0))
    fig.add_trace(go.Bar(name="Variance",  x=names, y=vars_,   marker_color=BLU, marker_line_width=0))
    fig.update_layout(**cl(title=ttl("Bias² vs Variance Decomposition"), height=280,
                           barmode="group"))
    return fig

def fig_walk_forward(wf_df):
    """Plot walk-forward MAPE per fold per model."""
    if wf_df.empty: return None
    fig = go.Figure()
    for col in wf_df.columns:
        if "_MAPE" not in col: continue
        name = col.replace("_MAPE","")
        fig.add_trace(go.Scatter(x=wf_df["fold"], y=wf_df[col], mode="lines+markers",
                                 name=name, line=dict(color=mc(name), width=2),
                                 marker=dict(size=7)))
    fig.update_layout(**cl(title=ttl("Walk-Forward MAPE by Fold"),height=300))
    return fig

# ── UI components ─────────────────────────────────────────────────────────────
_P = f"background:{BG1};border:1px solid {BD};border-radius:6px;padding:1rem 1.2rem;margin-bottom:0.75rem;"

def card(content, border_left=None):
    ex = f"border-left:3px solid {border_left};" if border_left else ""
    st.markdown(f'<div style="{_P}{ex}">{content}</div>', unsafe_allow_html=True)

def divider(label):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin:1.75rem 0 0.75rem;">'
        f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};white-space:nowrap;">{label}</span>'
        f'<div style="flex:1;height:1px;background:{BD};"></div></div>',
        unsafe_allow_html=True)

def stat_row(k,v):
    return (f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0;'
            f'border-bottom:1px solid {BD};font-size:0.8rem;">'
            f'<span style="color:{CDM};">{k}</span>'
            f'<span style="font-family:monospace;color:{CR};">{v}</span></div>')

def analyst_note(content, head="Analyst Note"):
    st.markdown(
        f'<div style="background:{BG2};border:1px solid {BD2};border-left:3px solid {G};'
        f'border-radius:5px;padding:0.85rem 1.1rem;margin:0.75rem 0;font-size:0.82rem;color:{CDM};line-height:1.7;">'
        f'<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;'
        f'color:{G};margin-bottom:0.35rem;">◈ {head}</div>{content}</div>',
        unsafe_allow_html=True)

def signal_box(kind, icon, text):
    bg={"bull":"rgba(62,207,142,0.08)","bear":"rgba(242,95,92,0.08)","neut":BG2,"caut":"rgba(240,165,0,0.07)"}
    bd={"bull":"rgba(62,207,142,0.2)","bear":"rgba(242,95,92,0.2)","neut":BD,"caut":"rgba(240,165,0,0.18)"}
    st.markdown(
        f'<div style="background:{bg[kind]};border:1px solid {bd[kind]};border-radius:5px;'
        f'padding:0.7rem 0.95rem;margin:0.35rem 0;font-size:0.8rem;display:flex;align-items:flex-start;gap:0.5rem;">'
        f'<span style="font-size:0.95rem;flex-shrink:0;">{icon}</span>'
        f'<span style="color:{CDM};line-height:1.5;">{text}</span></div>',
        unsafe_allow_html=True)

def empty_state(msg, sub=""):
    st.markdown(
        f'<div style="text-align:center;padding:4rem 2rem;">'
        f'<div style="font-size:2rem;opacity:0.25;margin-bottom:0.9rem;">◈</div>'
        f'<div style="font-size:0.88rem;color:{CDM};">{msg}</div>'
        f'<div style="font-size:0.74rem;color:{CMT};margin-top:0.35rem;">{sub}</div></div>',
        unsafe_allow_html=True)

def kpi_bar(cells):
    cols=st.columns(len(cells))
    for col,(lbl,val,sub,subc,accent) in zip(cols,cells):
        bdr=f"border-bottom:2px solid {G};" if accent else ""
        col.markdown(
            f'<div style="background:{BG1};border:1px solid {BD};border-radius:6px;'
            f'padding:1rem 1.1rem 0.9rem;{bdr}">'
            f'<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{CMT};margin-bottom:0.4rem;">{lbl}</div>'
            f'<div style="font-family:monospace;font-size:1.4rem;color:{CR};line-height:1;">{val}</div>'
            f'<div style="font-family:monospace;font-size:0.68rem;color:{subc};margin-top:0.3rem;">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

def model_status(name, state, detail=""):
    icons = {"wait":"○","run":"◌","done":"●","fail":"✕"}
    colors= {"wait":CMT,"run":AMB,"done":POS,"fail":NEG}
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;font-size:0.8rem;">'
        f'<span style="color:{colors[state]};font-size:0.9rem;">{icons[state]}</span>'
        f'<span style="color:{mc(name)};font-weight:600;">{name}</span>'
        f'<span style="color:{CDM};">{detail}</span></div>',
        unsafe_allow_html=True)

def to_csv(df): return df.to_csv(index=False).encode()

def to_excel(sheets):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        for name,df in sheets.items(): df.to_excel(w,sheet_name=name[:31],index=False)
    return buf.getvalue()

def sb_label(text):
    st.markdown(f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};margin:0.9rem 0 0.2rem;">{text}</p>',unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(
            f'<div style="background:{BG2};border-bottom:1px solid {BD};padding:1.1rem 1rem 0.9rem;margin-bottom:0.5rem;">'
            f'<div style="font-size:1.15rem;font-weight:700;color:{CR};">FinCast <span style="color:{G};">Pro</span>'
            f'<span style="font-size:0.6rem;color:{CMT};font-weight:400;margin-left:0.5rem;">v2</span></div>'
            f'<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.11em;text-transform:uppercase;color:{CMT};margin-top:0.1rem;">Institutional Forecasting</div>'
            f'</div>', unsafe_allow_html=True)

        # ── Data source ──────────────────────────────────────────────
        sb_label("Data Source")
        mode = st.radio("Data Source",
                        ["Upload CSV", "Yahoo Finance", "Sample Data"],
                        label_visibility="collapsed")

        df = None; freq = "MS"; freq_lbl = "Monthly"; auto_freq = False

        if mode == "Upload CSV":
            f = st.file_uploader("CSV with ds, y columns", type=["csv"],
                                 label_visibility="collapsed")
            if f:
                try:
                    raw = pd.read_csv(f)
                    # Flexible column mapping
                    col_map = {}
                    for c in raw.columns:
                        if c.lower() in ("ds","date","time","timestamp","period"): col_map[c]="ds"
                        elif c.lower() in ("y","value","close","price","revenue","sales"): col_map[c]="y"
                    if "ds" not in col_map.values() or "y" not in col_map.values():
                        # Use first two columns
                        cols=raw.columns.tolist()
                        col_map={cols[0]:"ds",cols[1]:"y"}
                        st.caption(f"Mapped: {cols[0]}→ds, {cols[1]}→y")
                    raw = raw.rename(columns=col_map)[["ds","y"]]
                    raw["ds"] = pd.to_datetime(raw["ds"])
                    raw["y"]  = pd.to_numeric(raw["y"], errors="coerce")
                    raw = raw.dropna().sort_values("ds").reset_index(drop=True)
                    df  = raw
                    freq, freq_lbl = auto_detect_freq(df)
                    auto_freq = True
                    st.success(f"✓ {len(df):,} rows · auto-detected: {freq_lbl}")
                except Exception as e:
                    st.error(f"Parse error: {e}")

        elif mode == "Yahoo Finance":
            ticker = st.text_input("Ticker symbol", value="AAPL",
                                   placeholder="AAPL, BTC-USD, MSFT…").upper().strip()
            period = st.selectbox("Period", ["1y","2y","5y","10y"], index=2)
            if ticker:
                with st.spinner(f"Fetching {ticker}…"):
                    df, err = fetch_yahoo(ticker, period)
                if err:
                    st.error(err)
                elif df is not None:
                    freq, freq_lbl = auto_detect_freq(df)
                    auto_freq = True
                    st.success(f"✓ {ticker} · {len(df):,} rows · {freq_lbl}")

        else:
            sample = st.selectbox("Dataset", ["Monthly Revenue","Stock / Asset Price"])
            df = gen_revenue() if "Revenue" in sample else gen_stock()
            freq = "MS" if "Revenue" in sample else "B"
            freq_lbl = "Monthly" if "Revenue" in sample else "Daily (Biz)"
            st.caption(f"Sample · {len(df):,} rows")

        if df is None: st.stop()

        # ── Frequency (shows detected value, allow override) ─────────
        sb_label("Frequency")
        freq_map = {"Monthly":"MS","Quarterly":"QS","Weekly":"W","Daily (Biz)":"B"}
        freq_lbl_sel = st.selectbox("Frequency",
                                    list(freq_map.keys()),
                                    index=list(freq_map.keys()).index(freq_lbl) if freq_lbl in freq_map else 0,
                                    label_visibility="collapsed")
        freq = freq_map[freq_lbl_sel]
        if auto_freq:
            st.caption(f"Auto-detected · override above if wrong")

        max_h = {"MS":24,"QS":8,"W":52,"B":252}[freq]
        def_h = {"MS":12,"QS":4,"W":26,"B":63}[freq]
        horizon = st.slider("Forecast Horizon", 1, max_h, min(def_h, max_h))

        # ── Pre-processing ───────────────────────────────────────────
        sb_label("Pre-Processing")
        winsor = st.checkbox("Winsorise outliers before fitting", value=False)
        st.caption("Clips anomalies to IQR bounds before model training")

        # ── Models ───────────────────────────────────────────────────
        sb_label("Models")
        c1,c2=st.columns(2)
        with c1:
            use_p=st.checkbox("Prophet",  value=True)
            use_x=st.checkbox("XGBoost",  value=True)
        with c2:
            use_a=st.checkbox("ARIMA",    value=True)
            use_m=st.checkbox("Monte Carlo", value=True)
        n_sims=st.select_slider("MC Simulations",[500,1000,2000,5000],value=1000) if use_m else 1000

        # ── Scenario & display ───────────────────────────────────────
        sb_label("Scenario")
        scenario=st.select_slider("Scenario",["Stress","Bear","Base","Bull","Upside"],value="Base")
        mc_scen={"Stress":"worst","Bear":"worst","Base":"base","Bull":"best","Upside":"best"}[scenario]

        sb_label("Display")
        show_ci      = st.checkbox("Show confidence intervals", value=True)
        show_regimes = st.checkbox("Show regime breaks", value=True)

        sb_label("Anomaly Detection")
        iqr_factor=st.slider("IQR Factor",1.0,4.0,2.0,0.5)

        st.divider()
        run=st.button("▶  RUN FORECAST ENGINE", use_container_width=True, type="primary")

        return (df, freq, horizon, use_p, use_a, use_x, use_m,
                n_sims, mc_scen, scenario, iqr_factor, run,
                winsor, show_ci, show_regimes)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    (df, freq, horizon, use_p, use_a, use_x, use_m,
     n_sims, mc_scen, scenario, iqr_factor, run,
     winsor, show_ci, show_regimes) = sidebar()

    # Anomalies on raw data (for display)
    s = df["y"]
    mask, lo, hi = detect_anomalies(s, iqr_factor)
    n_an = int(mask.sum())
    regimes = detect_regimes(s)

    # Winsorise for model fitting if toggled
    df_fit = winsorise(df, iqr_factor) if winsor else df

    # KPI metrics
    p_chg  = (s.iloc[-1]-s.iloc[0])/s.iloc[0]*100
    r_chg  = (s.iloc[-1]-s.iloc[-2])/s.iloc[-2]*100 if len(s)>1 else 0
    vol    = s.pct_change().std()*100
    _cagr  = cagr(s,freq)
    _mdd   = max_dd(s)
    _sh    = sharpe(s,freq)

    # Top bar
    now=datetime.now()
    st.markdown(
        f'<div style="background:{BG1};border:1px solid {BD};border-radius:8px;'
        f'padding:0.8rem 1.4rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">'
        f'<div><span style="font-size:1.3rem;font-weight:700;color:{CR};">FinCast</span>'
        f'<span style="font-size:1.3rem;font-weight:700;color:{G};"> Pro</span>'
        f'<span style="font-size:0.55rem;color:{CMT};border-left:1px solid {BD2};'
        f'padding-left:0.6rem;margin-left:0.7rem;letter-spacing:0.12em;text-transform:uppercase;">Institutional Forecasting</span></div>'
        f'<div style="font-family:monospace;font-size:0.68rem;color:{CMT};text-align:right;line-height:1.6;">'
        f'{now.strftime("%d %b %Y  %H:%M UTC")}<br>{len(df):,} obs · {freq} · {horizon}p · {scenario}'
        f'{"  · winsorised" if winsor else ""}</div></div>',
        unsafe_allow_html=True)

    kpi_bar([
        ("Latest Value",   f"{s.iloc[-1]:,.0f}",
         f"{'▲' if r_chg>=0 else '▼'} {abs(r_chg):.2f}% prior",
         POS if r_chg>=0 else NEG, True),
        ("Period Return",  f"{p_chg:+.1f}%",
         f"{len(s):,} observations",
         POS if p_chg>=0 else NEG, False),
        ("CAGR",           f"{_cagr:.1f}%", "Annualised", CDM, False),
        ("Max Drawdown",   f"{_mdd:.1f}%",  "Peak-to-trough",
         NEG if _mdd<-10 else CDM, False),
        ("Sharpe Ratio",   f"{_sh:.2f}",
         f"{'⚠ '+str(n_an)+' anom.' if n_an else '✓ Clean'}",
         AMB if n_an else POS, False),
    ])

    t1,t2,t3,t4,t5 = st.tabs([
        "📊 Data Intelligence",
        "🔮 Forecast Engine",
        "🎲 Monte Carlo",
        "📐 Validation",
        "📤 Export",
    ])

    # ══ TAB 1 ═══════════════════════════════════════════════════════
    with t1:
        divider("Historical Series")
        st.plotly_chart(fig_hist(df, mask, regimes, show_regimes), use_container_width=True)

        if show_regimes and regimes:
            signal_box("caut","◈",
                f"<strong>{len(regimes)} structural break{'s' if len(regimes)>1 else ''} detected</strong> via CUSUM. "
                "Regime changes may require retraining on the most recent segment only.")

        ca,cb=st.columns([3,2])
        with ca:
            divider("Year-over-Year Growth")
            st.plotly_chart(fig_yoy(df,freq), use_container_width=True)
        with cb:
            divider("Descriptive Statistics")
            rows="".join([
                stat_row("Observations",  f"{len(s):,}"),
                stat_row("Mean",          f"{s.mean():,.2f}"),
                stat_row("Median",        f"{s.median():,.2f}"),
                stat_row("Std Dev",       f"{s.std():,.2f}"),
                stat_row("Min / Max",     f"{s.min():,.0f} / {s.max():,.0f}"),
                stat_row("Skewness",      f"{s.skew():.4f}"),
                stat_row("Kurtosis",      f"{s.kurt():.4f}"),
                stat_row("Volatility σ",  f"{vol:.2f}%"),
                stat_row("CAGR",          f"{_cagr:.2f}%"),
                stat_row("Max Drawdown",  f"{_mdd:.2f}%"),
                stat_row("Sharpe Ratio",  f"{_sh:.3f}"),
                stat_row("Anomalies",     f"{n_an}"),
                stat_row("Regime Breaks", f"{len(regimes)}"),
            ])
            card(rows)

        cc,cd=st.columns([2,3])
        with cc:
            divider("Return Distribution")
            st.plotly_chart(fig_returns(s), use_container_width=True)
        with cd:
            divider(f"Anomaly Flags · {n_an} detected")
            if n_an==0:
                signal_box("bull","✓","<strong>No anomalies detected.</strong> Data quality passed.")
            else:
                adf=df[mask].copy(); adf["pct_dev"]=(adf["y"]-s.mean())/s.mean()*100
                for _,r in adf.iterrows():
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;padding:0.4rem 0.8rem;'
                        f'background:rgba(242,95,92,0.06);border-left:2px solid {NEG};margin:0.15rem 0;'
                        f'border-radius:3px;font-family:monospace;font-size:0.75rem;">'
                        f'<span style="color:{CMT};">{r["ds"].strftime("%Y-%m-%d")}</span>'
                        f'<span style="color:{CR};">{r["y"]:,.0f}</span>'
                        f'<span style="color:{NEG};">{r["pct_dev"]:+.1f}% vs μ</span></div>',
                        unsafe_allow_html=True)
                if winsor:
                    signal_box("bull","✦",
                        f"<strong>Winsorisation active.</strong> {n_an} outlier(s) clipped to IQR bounds before model fitting.")

        skew_txt=("positive skew — upside tail" if s.skew()>0.5
                  else "negative skew — downside tail" if s.skew()<-0.5
                  else "near-symmetric")
        analyst_note(
            f"Series shows <strong>{skew_txt}</strong>. CAGR <strong>{_cagr:.1f}%</strong>, "
            f"volatility <strong>{vol:.1f}%</strong>, Sharpe <strong>{_sh:.2f}</strong>"
            +(" — strong risk-adjusted return." if _sh>=1
              else " — adequate." if _sh>=0 else " — below risk-free rate.")+
            f" Max drawdown <strong>{_mdd:.1f}%</strong>."
            +(f" <strong>{n_an} anomal{'y' if n_an==1 else 'ies'}</strong> detected." if n_an else " Clean data.")
            +(f" <strong>{len(regimes)} regime break(s)</strong> detected." if regimes else ""),
            head="Data Intelligence Summary")

    # ══ TAB 2 ═══════════════════════════════════════════════════════
    with t2:
        if not run:
            empty_state("Configure parameters and click  ▶ RUN FORECAST ENGINE",
                        "Select models · Set horizon · Choose scenario")
            return

        fcs,ins_d={},{}; mc_paths=mc_fwd=None

        # Per-model status panel
        divider("Running Models")
        status_slots = {}
        model_list = (["Prophet"]*use_p+["ARIMA"]*use_a+
                      ["XGBoost"]*use_x+["Monte Carlo"]*use_m)
        cols_st = st.columns(max(1,len(model_list)))
        for i,name in enumerate(model_list):
            with cols_st[i]:
                status_slots[name]=st.empty()
                status_slots[name].markdown(
                    f'<div style="text-align:center;padding:0.6rem;background:{BG2};'
                    f'border:1px solid {BD};border-radius:5px;font-size:0.75rem;">'
                    f'<div style="color:{mc(name)};font-weight:700;">{name}</div>'
                    f'<div style="color:{CMT};">Queued</div></div>', unsafe_allow_html=True)

        def update_status(name, state, detail=""):
            icons={"run":"◌  Running…","done":"●  Done","fail":"✕  Failed"}
            colors={"run":AMB,"done":POS,"fail":NEG}
            status_slots[name].markdown(
                f'<div style="text-align:center;padding:0.6rem;background:{BG2};'
                f'border:1px solid {BD};border-radius:5px;font-size:0.75rem;">'
                f'<div style="color:{mc(name)};font-weight:700;">{name}</div>'
                f'<div style="color:{colors[state]};">{icons[state]}</div>'
                f'{"<div style=\"color:"+CMT+";font-size:0.65rem;\">"+detail+"</div>" if detail else ""}'
                f'</div>', unsafe_allow_html=True)

        df_hash = hash(str(df_fit["y"].values.tolist())+str(horizon)+freq)

        if use_p:
            update_status("Prophet","run")
            fwd,ins=run_prophet(df_hash,df_fit,horizon,freq)
            if fwd is not None:
                fcs["Prophet"]=fwd; ins_d["Prophet"]=ins
                m=mape(df_fit.merge(ins,on="ds")["y"].values, df_fit.merge(ins,on="ds")["y_pred"].values)
                update_status("Prophet","done",f"MAPE {m:.1f}%")
            else: update_status("Prophet","fail","install prophet")

        if use_a:
            update_status("ARIMA","run")
            fwd,ins=run_arima(df_hash,df_fit,horizon,freq)
            if fwd is not None:
                fcs["ARIMA"]=fwd; ins_d["ARIMA"]=ins
                m=mape(df_fit.merge(ins,on="ds")["y"].values, df_fit.merge(ins,on="ds")["y_pred"].values)
                update_status("ARIMA","done",f"MAPE {m:.1f}%")
            else: update_status("ARIMA","fail","install statsmodels")

        if use_x:
            update_status("XGBoost","run")
            fwd,ins=run_xgboost(df_hash,df_fit,horizon,freq)
            if fwd is not None:
                fcs["XGBoost"]=fwd; ins_d["XGBoost"]=ins
                m=mape(df_fit.merge(ins,on="ds")["y"].values, df_fit.merge(ins,on="ds")["y_pred"].values)
                update_status("XGBoost","done",f"MAPE {m:.1f}%")
            else: update_status("XGBoost","fail","install xgboost")

        if use_m:
            update_status("Monte Carlo","run")
            mc_fwd,mc_paths=run_monte_carlo(df_hash,df_fit,horizon,freq,n_sims,mc_scen)
            if mc_fwd is not None:
                fcs["Monte Carlo"]=mc_fwd
                update_status("Monte Carlo","done",f"{n_sims:,} sims")
            else: update_status("Monte Carlo","fail")

        if not fcs: st.error("No models ran."); return

        ens=build_ensemble(fcs)
        st.session_state.update({"fcs":fcs,"ins_d":ins_d,"mc_paths":mc_paths,
                                  "mc_fwd":mc_fwd,"ens":ens,"df_fit":df_fit})

        divider("Multi-Model Forecast")
        st.plotly_chart(fig_forecast(df,fcs,ens,show_ci), use_container_width=True)

        divider("Terminal Estimates")
        all_models={**fcs,**({"Ensemble":ens} if ens is not None else {})}
        kpi_items=[]
        for name,fwd in all_models.items():
            t_val=float(fwd["yhat"].iloc[-1]); s_last=float(s.iloc[-1])
            t_chg=(t_val-s_last)/s_last*100
            kpi_items.append((name,f"{t_val:,.0f}",
                               ("▲" if t_chg>=0 else "▼")+f" {abs(t_chg):.1f}%",
                               POS if t_chg>=0 else NEG, name=="Ensemble"))
        kpi_bar(kpi_items)

        divider("Forecast Table")
        rows=[]
        for d in fcs[list(fcs.keys())[0]]["ds"]:
            row={"Period":d.strftime("%Y-%m-%d")}
            for name,fwd in fcs.items():
                m2=fwd[fwd["ds"]==d]
                if not m2.empty:
                    row[name]=round(float(m2["yhat"].values[0]),2)
                    row[f"{name} Lo"]=round(float(m2["lower"].values[0]),2)
                    row[f"{name} Hi"]=round(float(m2["upper"].values[0]),2)
            if ens is not None:
                em=ens[ens["ds"]==d]
                if not em.empty: row["Ensemble"]=round(float(em["yhat"].values[0]),2)
            rows.append(row)
        tbl=pd.DataFrame(rows)
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.session_state["tbl"]=tbl

        divider("Forecast Signals")
        if ens is not None:
            ec=float((ens["yhat"].iloc[-1]-s.iloc[-1])/s.iloc[-1]*100)
            if   ec>5:  signal_box("bull","▲",f"<strong>Bullish:</strong> Consensus projects <strong>+{ec:.1f}%</strong> over {horizon} periods ({scenario}).")
            elif ec<-5: signal_box("bear","▼",f"<strong>Bearish:</strong> Consensus projects <strong>{ec:.1f}%</strong> over {horizon} periods.")
            else:       signal_box("neut","◆",f"<strong>Neutral:</strong> Ensemble projects <strong>{ec:+.1f}%</strong> movement.")
        if winsor and n_an>0:
            signal_box("bull","✦",f"<strong>Winsorisation applied:</strong> {n_an} outlier(s) treated before fitting. Forecasts may differ materially from raw-data run.")
        if n_an>0 and not winsor:
            signal_box("caut","⚠",f"<strong>Data Alert:</strong> {n_an} anomal{'y' if n_an==1 else 'ies'} in training data — consider enabling winsorisation.")

    # ══ TAB 3 ═══════════════════════════════════════════════════════
    with t3:
        if not run: empty_state("Run the forecast engine first."); return
        mc_paths=st.session_state.get("mc_paths"); mc_fwd=st.session_state.get("mc_fwd")
        if not use_m or mc_paths is None: st.info("Enable Monte Carlo in the sidebar."); return

        divider(f"Monte Carlo Fan · {n_sims:,} Simulations · {scenario}")
        st.plotly_chart(fig_mc_fan(df,mc_paths,mc_fwd["ds"]), use_container_width=True)

        ca,cb=st.columns([3,2])
        with ca:
            divider("Terminal Value Distribution")
            st.plotly_chart(fig_terminal(mc_paths), use_container_width=True)
        with cb:
            tv=mc_paths[:,-1]; var95=np.percentile(tv,5); cvar95=tv[tv<=var95].mean()
            p_loss=np.mean(tv<float(s.iloc[-1]))*100; p_10up=np.mean(tv>float(s.iloc[-1])*1.1)*100
            divider("Risk Metrics")
            card("".join([
                stat_row("VaR (95%)",    f"{var95:,.0f}"),
                stat_row("CVaR / ES",    f"{cvar95:,.0f}"),
                stat_row("P(loss)",      f"{p_loss:.1f}%"),
                stat_row("P(gain>10%)",  f"{p_10up:.1f}%"),
                stat_row("Median",       f"{np.percentile(tv,50):,.0f}"),
                stat_row("P10 / P90",    f"{np.percentile(tv,10):,.0f} / {np.percentile(tv,90):,.0f}"),
                stat_row("Scenario",     scenario),
                stat_row("Simulations",  f"{n_sims:,}"),
            ]))
            divider("Percentile Table")
            st.dataframe(pd.DataFrame({
                "Pctl": [f"P{p}" for p in [1,5,10,25,50,75,90,95,99]],
                "Value":[f"{np.percentile(tv,p):,.0f}" for p in [1,5,10,25,50,75,90,95,99]],
                "vs Last":[f"{(np.percentile(tv,p)/float(s.iloc[-1])-1)*100:+.1f}%" for p in [1,5,10,25,50,75,90,95,99]],
            }), hide_index=True, use_container_width=True)

        analyst_note(
            f"<strong>{n_sims:,} GBM simulations</strong> under <strong>{scenario}</strong>. "
            f"VaR (95%): <strong>{var95:,.0f}</strong> · CVaR: <strong>{cvar95:,.0f}</strong>. "
            f"P(loss): <strong>{p_loss:.1f}%</strong> · P(+10%): <strong>{p_10up:.1f}%</strong>.",
            head="Risk Summary")

    # ══ TAB 4 ═══════════════════════════════════════════════════════
    with t4:
        if not run: empty_state("Run the forecast engine first."); return
        ins_d=st.session_state.get("ins_d",{})
        df_fit2=st.session_state.get("df_fit",df)
        if not ins_d: st.info("Select at least one parametric model."); return

        scores={}
        for name,ins in ins_d.items():
            merged=df_fit2.merge(ins,on="ds",how="inner")
            if merged.empty: continue
            a,f_=merged["y"].values,merged["y_pred"].values
            scores[name]={"MAPE":round(mape(a,f_),3),"sMAPE":round(smape(a,f_),3),
                          "RMSE":round(rmse(a,f_),2)}
        st.session_state["scores"]=scores
        if not scores: st.warning("Could not compute scores."); return

        ranked=sorted(scores,key=lambda m:scores[m]["MAPE"])
        rank_icons=["🥇","🥈","🥉"]

        divider("Model Leaderboard")
        lb_rows=[]
        for i,name in enumerate(ranked):
            sc=scores[name]
            lb_rows.append({
                "Rank": rank_icons[i] if i<3 else str(i+1),
                "Model": name+("  ✦ Best Fit" if i==0 else ""),
                "MAPE %": f"{sc['MAPE']:.3f}",
                "sMAPE %": f"{sc['sMAPE']:.3f}",
                "RMSE": f"{sc['RMSE']:,.2f}",
            })
        st.dataframe(pd.DataFrame(lb_rows), hide_index=True, use_container_width=True)

        divider("Accuracy Comparison")
        st.plotly_chart(fig_accuracy(scores), use_container_width=True)

        divider("In-Sample Fit vs Actual")
        st.plotly_chart(fig_fit(df_fit2,ins_d), use_container_width=True)

        # ── Walk-forward backtesting ──────────────────────────────────
        divider("Walk-Forward Backtesting · Rolling OOS Validation")
        wf_hash = hash(str(df_fit2["y"].values.tolist())+freq)
        with st.spinner("Running walk-forward splits…"):
            wf_df = run_walk_forward(wf_hash, df_fit2, freq,
                                     n_splits=5, use_p=use_p, use_a=use_a, use_x=use_x)
        if not wf_df.empty:
            wf_fig = fig_walk_forward(wf_df)
            if wf_fig: st.plotly_chart(wf_fig, use_container_width=True)
            # Summary table
            display_cols = ["fold","train_n","test_n"]
            mape_cols = [c for c in wf_df.columns if "_MAPE" in c]
            rmse_cols = [c for c in wf_df.columns if "_RMSE" in c]
            st.dataframe(wf_df[display_cols+mape_cols+rmse_cols],
                         hide_index=True, use_container_width=True)
            # Compute average OOS MAPE per model
            for mc_ in mape_cols:
                nm=mc_.replace("_MAPE","")
                avg=wf_df[mc_].mean()
                signal_box("neut","◈",
                    f"<strong>{nm}</strong> avg OOS MAPE: <strong>{avg:.2f}%</strong> across {len(wf_df)} folds")
        else:
            st.caption("Not enough data for walk-forward splits (need ≥48 observations).")

        # ── Residual diagnostics ─────────────────────────────────────
        divider("Residual Diagnostics · ACF")
        rf = fig_residuals(df_fit2, ins_d)
        if rf: st.plotly_chart(rf, use_container_width=True)

        # ── Bias / variance decomposition ────────────────────────────
        divider("Bias² vs Variance Decomposition")
        bv_fig = fig_bias_variance(scores, ins_d, df_fit2)
        if bv_fig:
            st.plotly_chart(bv_fig, use_container_width=True)
            analyst_note(
                "High <strong>Bias²</strong> (red) means the model is systematically off — consider a richer model or more features. "
                "High <strong>Variance</strong> (blue) means the model is overfitting — reduce complexity or use more training data.",
                head="Bias–Variance Guide")

        best=ranked[0]
        gap_txt=""
        if len(ranked)>1:
            gap=abs(scores[ranked[0]]["MAPE"]-scores[ranked[1]]["MAPE"])
            gap_txt=f" Gap to runner-up: <strong>{gap:.2f}pp</strong> — "+(
                "ensemble recommended." if gap<1 else "prefer the top model.")
        analyst_note(
            f"<strong>{best}</strong> achieves in-sample MAPE <strong>{scores[best]['MAPE']:.2f}%</strong>, "
            f"RMSE <strong>{scores[best]['RMSE']:,.2f}</strong>.{gap_txt} "
            "Walk-forward OOS MAPE above reflects true out-of-sample accuracy on held-out folds.",
            head="Validation Guidance")

    # ══ TAB 5 ═══════════════════════════════════════════════════════
    with t5:
        divider("Download Outputs")
        ds_str=datetime.today().strftime("%Y%m%d")
        tbl=st.session_state.get("tbl")
        hist_exp=df.copy(); hist_exp["anomaly_flag"]=mask.astype(int)

        c1,c2,c3=st.columns(3)
        with c1:
            card(f'<div style="font-weight:700;color:{CR};margin-bottom:0.3rem;">📄 Forecast Table — CSV</div>'
                 f'<div style="font-size:0.75rem;color:{CDM};margin-bottom:0.8rem;">All models + CI. Excel / Tableau / PowerBI ready.</div>')
            if tbl is not None:
                st.download_button("↓ Download Forecast CSV", to_csv(tbl),
                                   f"FinCast_Forecast_{ds_str}.csv","text/csv")
            else: st.caption("Run forecast first.")
        with c2:
            card(f'<div style="font-weight:700;color:{CR};margin-bottom:0.3rem;">📊 Historical Data — CSV</div>'
                 f'<div style="font-size:0.75rem;color:{CDM};margin-bottom:0.8rem;">Cleaned series with anomaly flags.</div>')
            st.download_button("↓ Download Historical CSV", to_csv(hist_exp),
                               f"FinCast_Historical_{ds_str}.csv","text/csv")
        with c3:
            card(f'<div style="font-weight:700;color:{CR};margin-bottom:0.3rem;">📁 Full Report — Excel</div>'
                 f'<div style="font-size:0.75rem;color:{CDM};margin-bottom:0.8rem;">Historical · Forecast · Scores · Walk-Forward · MC.</div>')
            if tbl is not None:
                sheets={"Historical":hist_exp,"Forecast":tbl}
                sc=st.session_state.get("scores",{})
                if sc:
                    sdf=pd.DataFrame(sc).T.reset_index(); sdf.columns=["Model","MAPE","sMAPE","RMSE"]
                    sheets["Model Accuracy"]=sdf
                wf=st.session_state.get("wf_df")
                if wf is not None and not wf.empty: sheets["Walk-Forward"]=wf
                mc_fwd2=st.session_state.get("mc_fwd")
                if mc_fwd2 is not None: sheets["MC Summary"]=mc_fwd2[["ds","yhat","lower","upper"]]
                st.download_button("↓ Download Excel Report",to_excel(sheets),
                                   f"FinCast_Report_{ds_str}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.caption("Run forecast first.")

        if tbl is not None:
            sc=st.session_state.get("scores",{})
            best_m=min(sc,key=lambda m:sc[m]["MAPE"]) if sc else "N/A"
            ens2=st.session_state.get("ens")
            ens_t=f"{float(ens2['yhat'].iloc[-1]):,.0f}" if ens2 is not None else "N/A"
            analyst_note(
                f"Generated <strong>{datetime.now().strftime('%d %b %Y, %H:%M UTC')}</strong>. "
                f"<strong>{len(df):,} obs</strong> · {freq} · {horizon}-period horizon · {scenario}. "
                f"Best model: <strong>{best_m}</strong>"+(f" (MAPE {sc[best_m]['MAPE']:.2f}%)" if best_m!="N/A" else "")+
                f". Ensemble terminal: <strong>{ens_t}</strong>. "
                f"CAGR <strong>{_cagr:.1f}%</strong> · MDD <strong>{_mdd:.1f}%</strong> · Sharpe <strong>{_sh:.2f}</strong>"
                +(f" · <strong>Winsorised</strong>" if winsor else "")+
                f" · {len(regimes)} regime break(s).",
                head="Report Summary")

if __name__=="__main__":
    main()
