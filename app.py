"""
FinCast Pro — Institutional Financial Forecasting Platform
Zero external CSS — all inline styles. Dark theme via config.toml.
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

st.set_page_config(
    page_title="FinCast Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour tokens (used in inline styles & charts) ─────────────────────────
G   = "#c8a951"   # gold
CR  = "#e8e2d5"   # cream
CDM = "#9e9888"   # cream-dim
CMT = "#4a4a5e"   # cream-muted
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

MC = {"Prophet": G, "ARIMA": BLU, "XGBoost": POS, "Monte Carlo": NEG, "Ensemble": PRP}

def mc(name): return MC.get(name, CDM)

# ── Chart base ──────────────────────────────────────────────────────────────
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
    d = copy.deepcopy(_BL); d.update(kw); return d

def ttl(t):
    return dict(text=t, font=dict(color=CDM, size=12), x=0.01)

# ── Data helpers ────────────────────────────────────────────────────────────
def gen_revenue(n=48):
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today().replace(day=1), periods=n, freq="MS")
    trend = np.linspace(480_000, 880_000, n)
    season = 75_000 * np.sin(2*np.pi*np.arange(n)/12 - np.pi/3)
    vals = np.clip(trend + season + np.random.normal(0, 22_000, n), 0, None)
    vals[n-4] *= 0.14
    return pd.DataFrame({"ds": dates, "y": vals.round(0)})

def gen_stock(n=500):
    np.random.seed(7)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    price = 120 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
    return pd.DataFrame({"ds": dates, "y": price.round(4)})

def detect_anomalies(s, factor=2.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1-factor*iqr) | (s > q3+factor*iqr), q1-factor*iqr, q3+factor*iqr

def cagr(s, freq):
    ppy = {"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    n = len(s)/ppy
    if n<=0 or s.iloc[0]<=0: return 0.0
    return ((s.iloc[-1]/s.iloc[0])**(1/n)-1)*100

def max_dd(s):
    return ((s - s.cummax())/s.cummax()).min()*100

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

# ── Models ──────────────────────────────────────────────────────────────────
def _off(freq):
    try: return pd.tseries.frequencies.to_offset(freq)
    except: return pd.DateOffset(months=1)

def run_prophet(df, horizon, freq):
    try:
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, interval_width=0.90,
                    changepoint_prior_scale=0.05)
        m.fit(df[["ds","y"]])
        fut = m.make_future_dataframe(periods=horizon, freq=freq)
        fc  = m.predict(fut)
        fwd = fc[fc["ds"]>df["ds"].max()][["ds","yhat","yhat_lower","yhat_upper"]].copy()
        fwd.columns = ["ds","yhat","lower","upper"]
        ins = fc[fc["ds"]<=df["ds"].max()][["ds","yhat"]].rename(columns={"yhat":"y_pred"})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception: return None, None

def run_arima(df, horizon, freq):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        s = df.set_index("ds")["y"]
        sp = {"MS":12,"W":52,"B":5,"QS":4}.get(freq,12)
        seas = (1,1,0,sp) if len(s)>=2*sp else (0,0,0,0)
        res = SARIMAX(s, order=(1,1,1), seasonal_order=seas,
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=200)
        fc = res.get_forecast(steps=horizon)
        ci = fc.conf_int(alpha=0.10)
        dates = pd.date_range(start=df["ds"].max()+_off(freq), periods=horizon, freq=freq)
        fwd = pd.DataFrame({"ds":dates,"yhat":fc.predicted_mean.values,
                            "lower":ci.iloc[:,0].values,"upper":ci.iloc[:,1].values})
        ins = pd.DataFrame({"ds":s.index,"y_pred":res.fittedvalues.values})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception: return None, None

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
        X,ya=feat(s,n_lags); sp=max(1,int(len(X)*0.15))
        mdl=xgb.XGBRegressor(n_estimators=500,learning_rate=0.04,max_depth=4,
                               subsample=0.8,colsample_bytree=0.8,random_state=42,verbosity=0)
        mdl.fit(X[:-sp],ya[:-sp],eval_set=[(X[-sp:],ya[-sp:])],verbose=False)
        Xa,_=feat(s,n_lags)
        ins=pd.DataFrame({"ds":df["ds"].iloc[n_lags:].values,"y_pred":mdl.predict(Xa)})
        win=list(s[-n_lags:]); yhats,lows,highs=[],[],[]; ns=np.std(s)*0.04
        for step in range(horizon):
            w=np.array(win[-n_lags:])
            f_=np.array([*w,np.mean(w),np.std(w),np.min(w),np.max(w),w[-1]-w[0],w[-1]/(np.mean(w)+1e-9)]).reshape(1,-1)
            p=float(mdl.predict(f_)[0]); sp2=ns*np.sqrt(step+1)*1.645
            yhats.append(p); lows.append(p-sp2); highs.append(p+sp2); win.append(p)
        dates=pd.date_range(start=df["ds"].max()+_off(freq),periods=horizon,freq=freq)
        fwd=pd.DataFrame({"ds":dates,"yhat":yhats,"lower":lows,"upper":highs})
        return fwd.reset_index(drop=True), ins.reset_index(drop=True)
    except Exception: return None, None

def run_monte_carlo(df, horizon, freq, n_sims=1000, scenario="base"):
    s=df["y"].values.astype(float); lr=np.diff(np.log(s+1e-9)); mu,sig=np.mean(lr),np.std(lr)
    tm,ts={"best":(1.5,0.65),"base":(1.0,1.0),"worst":(0.3,1.5)}.get(scenario,(1.0,1.0))
    np.random.seed(99); last=s[-1]; paths=np.zeros((n_sims,horizon))
    for i in range(n_sims):
        paths[i]=last*np.exp(np.cumsum(np.random.normal(mu*tm,sig*ts,horizon)))
    dates=pd.date_range(start=df["ds"].max()+_off(freq),periods=horizon,freq=freq)
    fwd=pd.DataFrame({"ds":dates,"yhat":np.percentile(paths,50,axis=0),
                       "lower":np.percentile(paths,10,axis=0),"upper":np.percentile(paths,90,axis=0)})
    return fwd.reset_index(drop=True), paths

def build_ensemble(forecasts):
    valid={k:v for k,v in forecasts.items() if v is not None}
    if len(valid)<2: return None
    base=list(valid.values())[0][["ds"]].copy()
    base["yhat"]=np.mean([v["yhat"].values for v in valid.values()],axis=0)
    base["lower"]=np.mean([v["lower"].values for v in valid.values()],axis=0)
    base["upper"]=np.mean([v["upper"].values for v in valid.values()],axis=0)
    return base

# ── Charts ──────────────────────────────────────────────────────────────────
def fig_hist(df, mask, lo, hi):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
        line=dict(color=CDM,width=1.8),fill="tozeroy",fillcolor="rgba(107,107,130,0.05)"))
    if mask.any():
        adf=df[mask]
        fig.add_trace(go.Scatter(x=adf["ds"],y=adf["y"],mode="markers",name="Anomaly",
            marker=dict(color=NEG,size=10,symbol="circle-open",line=dict(width=2,color=NEG))))
    fig.add_hline(y=df["y"].mean(),line_dash="dot",line_color=BD2,line_width=1,
                  annotation_text=f"Mean {df['y'].mean():,.0f}",
                  annotation_font_color=CMT,annotation_font_size=10,annotation_position="bottom right")
    fig.update_layout(**cl(title=ttl("Historical Series · Anomaly Detection"),height=340))
    return fig

def fig_yoy(df, freq):
    ppy={"MS":12,"W":52,"B":252,"QS":4}.get(freq,12)
    yoy=df.set_index("ds")["y"].pct_change(ppy).dropna()*100
    colors=[POS if v>=0 else NEG for v in yoy]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=yoy.index,y=yoy.values,marker_color=colors,marker_line_width=0,name="YoY%"))
    fig.add_hline(y=0,line_color=BD2,line_width=1)
    fig.update_layout(**cl(title=ttl("Year-over-Year Growth (%)"),height=260,showlegend=False))
    return fig

def fig_returns(s):
    r=s.pct_change().dropna()*100
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=r,nbinsx=50,marker_color=f"rgba(200,169,81,0.65)",
                               marker_line_width=0,name="Returns"))
    fig.add_vline(x=r.mean(),line_dash="dash",line_color=POS,line_width=1.5,
                  annotation_text=f"μ={r.mean():.2f}%",annotation_font_color=POS,
                  annotation_font_size=10,annotation_position="top right")
    fig.add_vline(x=0,line_dash="solid",line_color=BD2,line_width=1)
    fig.update_layout(**cl(title=ttl("Return Distribution (%)"),height=290,showlegend=False))
    return fig

def fig_forecast(df, forecasts, ens):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Actual",
                             line=dict(color=CDM,width=2)))
    fig.add_vline(x=df["ds"].max(),line_dash="dot",line_color=BD2,line_width=1.5,
                  annotation_text="Forecast →",annotation_font_color=CMT,
                  annotation_font_size=10,annotation_position="top right")
    for name in ["Prophet","ARIMA","XGBoost","Monte Carlo"]:
        fwd=forecasts.get(name)
        if fwd is None: continue
        col=mc(name); r,g,b=int(col[1:3],16),int(col[3:5],16),int(col[5:7],16)
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
    fig.update_layout(**cl(title=ttl("Multi-Model Forecast · 90% Confidence Intervals"),height=420))
    return fig

def fig_mc_fan(df, paths, dates):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"],y=df["y"],mode="lines",name="Historical",
                             line=dict(color=CDM,width=2)))
    idx=np.random.choice(paths.shape[0],min(150,paths.shape[0]),replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(x=dates,y=paths[i],mode="lines",
            line=dict(color="rgba(242,95,92,0.04)",width=1),showlegend=False,hoverinfo="skip"))
    for p,lbl,col in [(90,"P90",NEG),(75,"P75",AMB),(50,"Median",CR),(25,"P25",POS),(10,"P10",BLU)]:
        fig.add_trace(go.Scatter(x=dates,y=np.percentile(paths,p,axis=0),mode="lines",name=lbl,
            line=dict(color=col,width=2 if lbl=="Median" else 1.2,
                      dash="solid" if lbl=="Median" else "dot")))
    fig.add_vline(x=df["ds"].max(),line_dash="dot",line_color=BD2,line_width=1.5)
    fig.update_layout(**cl(title=ttl("Monte Carlo Fan · Simulated Paths"),height=400))
    return fig

def fig_terminal(paths):
    t=paths[:,-1]
    fig=go.Figure()
    fig.add_trace(go.Histogram(x=t,nbinsx=70,marker_color="rgba(242,95,92,0.55)",
                               marker_line_width=0))
    for p,col,lbl in [(5,NEG,"P5 VaR"),(50,CR,"Median"),(95,POS,"P95")]:
        fig.add_vline(x=np.percentile(t,p),line_dash="dash",line_color=col,line_width=1.5,
                      annotation_text=lbl,annotation_font_color=col,
                      annotation_font_size=10,annotation_position="top")
    fig.update_layout(**cl(title=ttl("Terminal Value Distribution"),height=300,showlegend=False))
    return fig

def fig_accuracy(scores):
    models=list(scores); mapes=[scores[m]["MAPE"] for m in models]; rmses=[scores[m]["RMSE"] for m in models]
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

# ── Inline-style HTML components ────────────────────────────────────────────
_P  = f"background:{BG1};border:1px solid {BD};border-radius:6px;padding:1rem 1.2rem;margin-bottom:0.75rem;"
_PH = f"font-size:0.75rem;font-weight:700;color:{CR};margin-bottom:0.6rem;"

def card(content, border_left=None):
    extra = f"border-left:3px solid {border_left};" if border_left else ""
    st.markdown(f'<div style="{_P}{extra}">{content}</div>', unsafe_allow_html=True)

def divider(label):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.6rem;margin:1.75rem 0 0.75rem;">'
        f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};white-space:nowrap;">{label}</span>'
        f'<div style="flex:1;height:1px;background:{BD};"></div></div>',
        unsafe_allow_html=True)

def stat_row(k, v):
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
    bg_map = {"bull": f"rgba(62,207,142,0.08)", "bear": f"rgba(242,95,92,0.08)",
              "neut": BG2, "caut": "rgba(240,165,0,0.07)"}
    bd_map = {"bull": "rgba(62,207,142,0.2)", "bear": "rgba(242,95,92,0.2)",
              "neut": BD, "caut": "rgba(240,165,0,0.18)"}
    st.markdown(
        f'<div style="background:{bg_map[kind]};border:1px solid {bd_map[kind]};border-radius:5px;'
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
    """cells = list of (label, value, sub, sub_color, accent)"""
    cols = st.columns(len(cells))
    for col, (lbl, val, sub, subc, accent) in zip(cols, cells):
        border = f"border-bottom:2px solid {G};" if accent else ""
        col.markdown(
            f'<div style="background:{BG1};border:1px solid {BD};border-radius:6px;'
            f'padding:1rem 1.1rem 0.9rem;{border}">'
            f'<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.14em;'
            f'text-transform:uppercase;color:{CMT};margin-bottom:0.4rem;">{lbl}</div>'
            f'<div style="font-family:monospace;font-size:1.4rem;color:{CR};line-height:1;">{val}</div>'
            f'<div style="font-family:monospace;font-size:0.68rem;color:{subc};margin-top:0.3rem;">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

def to_csv(df): return df.to_csv(index=False).encode()

def to_excel(sheets):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        for name,df in sheets.items(): df.to_excel(w,sheet_name=name[:31],index=False)
    return buf.getvalue()

# ── Sidebar ──────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(
            f'<div style="background:{BG2};border-bottom:1px solid {BD};padding:1.1rem 1rem 0.9rem;margin-bottom:0.5rem;">'
            f'<div style="font-size:1.15rem;font-weight:700;color:{CR};">FinCast <span style="color:{G};">Pro</span></div>'
            f'<div style="font-size:0.6rem;font-weight:600;letter-spacing:0.11em;text-transform:uppercase;color:{CMT};margin-top:0.1rem;">Institutional Forecasting</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown(f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};margin:0.75rem 0 0.2rem;">Data Source</p>', unsafe_allow_html=True)
        mode = st.radio("Data Source", ["Upload CSV", "Use Sample Data"], label_visibility="collapsed")

        df = None
        if mode == "Upload CSV":
            f = st.file_uploader("CSV with columns ds, y", type=["csv"], label_visibility="collapsed")
            if f:
                try:
                    df = pd.read_csv(f, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
                    st.success(f"✓ {len(df):,} rows loaded")
                except Exception as e:
                    st.error(f"Parse error: {e}")
        else:
            sample = st.selectbox("Dataset", ["Monthly Revenue", "Stock / Asset Price"], label_visibility="visible")
            df = gen_revenue() if "Revenue" in sample else gen_stock()
            st.caption(f"Sample data · {len(df):,} rows")

        if df is None: st.stop()

        st.markdown(f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};margin:0.75rem 0 0.2rem;">Forecast Parameters</p>', unsafe_allow_html=True)
        freq_map = {"Monthly":"MS","Quarterly":"QS","Weekly":"W","Daily (Biz)":"B"}
        freq_lbl = st.selectbox("Frequency", list(freq_map.keys()))
        freq = freq_map[freq_lbl]
        max_h = {"MS":24,"QS":8,"W":52,"B":252}[freq]
        def_h = {"MS":12,"QS":4,"W":26,"B":63}[freq]
        horizon = st.slider("Forecast Horizon", 1, max_h, def_h)

        st.markdown(f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};margin:0.75rem 0 0.2rem;">Models</p>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            use_p = st.checkbox("Prophet",  value=True)
            use_x = st.checkbox("XGBoost",  value=True)
        with c2:
            use_a = st.checkbox("ARIMA",    value=True)
            use_m = st.checkbox("Monte Carlo", value=True)
        n_sims = st.select_slider("MC Simulations", [500,1000,2000,5000], value=1000) if use_m else 1000

        st.markdown(f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};margin:0.75rem 0 0.2rem;">Scenario</p>', unsafe_allow_html=True)
        scenario = st.select_slider("Scenario", ["Stress","Bear","Base","Bull","Upside"], value="Base")
        mc_scen = {"Stress":"worst","Bear":"worst","Base":"base","Bull":"best","Upside":"best"}[scenario]

        st.markdown(f'<p style="font-size:0.62rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:{G};margin:0.75rem 0 0.2rem;">Anomaly Detection</p>', unsafe_allow_html=True)
        iqr_factor = st.slider("IQR Factor", 1.0, 4.0, 2.0, 0.5)

        st.divider()
        run = st.button("▶  RUN FORECAST ENGINE", use_container_width=True, type="primary")

        return df, freq, horizon, use_p, use_a, use_x, use_m, n_sims, mc_scen, scenario, iqr_factor, run

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    df, freq, horizon, use_p, use_a, use_x, use_m, n_sims, mc_scen, scenario, iqr_factor, run = sidebar()

    # Top bar
    now = datetime.now()
    st.markdown(
        f'<div style="background:{BG1};border:1px solid {BD};border-radius:8px;'
        f'padding:0.8rem 1.4rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">'
        f'<div><span style="font-size:1.3rem;font-weight:700;color:{CR};">FinCast</span>'
        f'<span style="font-size:1.3rem;font-weight:700;color:{G};"> Pro</span>'
        f'<span style="font-size:0.6rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;'
        f'color:{CMT};border-left:1px solid {BD2};padding-left:0.6rem;margin-left:0.7rem;">Institutional Forecasting</span></div>'
        f'<div style="font-family:monospace;font-size:0.68rem;color:{CMT};text-align:right;line-height:1.6;">'
        f'{now.strftime("%d %b %Y  %H:%M UTC")}<br>{len(df):,} obs · {freq} · {horizon}p · {scenario}</div>'
        f'</div>', unsafe_allow_html=True)

    # KPIs
    s = df["y"]
    mask, lo, hi = detect_anomalies(s, iqr_factor)
    n_an = int(mask.sum())
    p_chg = (s.iloc[-1]-s.iloc[0])/s.iloc[0]*100
    r_chg = (s.iloc[-1]-s.iloc[-2])/s.iloc[-2]*100 if len(s)>1 else 0
    vol   = s.pct_change().std()*100
    _cagr = cagr(s,freq)
    _mdd  = max_dd(s)
    _sh   = sharpe(s,freq)

    kpi_bar([
        ("Latest Value",   f"{s.iloc[-1]:,.0f}",
         f"{'▲' if r_chg>=0 else '▼'} {abs(r_chg):.2f}% prior",
         POS if r_chg>=0 else NEG, True),
        ("Period Return",  f"{p_chg:+.1f}%",
         f"{len(s):,} observations",
         POS if p_chg>=0 else NEG, False),
        ("CAGR",           f"{_cagr:.1f}%",
         "Annualised growth", CDM, False),
        ("Max Drawdown",   f"{_mdd:.1f}%",
         "Peak-to-trough",
         NEG if _mdd<-10 else CDM, False),
        ("Sharpe Ratio",   f"{_sh:.2f}",
         f"{'⚠ ' + str(n_an) + ' anomal' + ('y' if n_an==1 else 'ies') if n_an else '✓ Clean data'}",
         AMB if n_an else POS, False),
    ])

    # Tabs
    t1,t2,t3,t4,t5 = st.tabs([
        "📊 Data Intelligence",
        "🔮 Forecast Engine",
        "🎲 Monte Carlo",
        "📐 Model Validation",
        "📤 Export",
    ])

    # ══ TAB 1 ══════════════════════════════════════════════════════════
    with t1:
        divider("Historical Series")
        st.plotly_chart(fig_hist(df,mask,lo,hi), use_container_width=True)

        ca,cb = st.columns([3,2])
        with ca:
            divider("Year-over-Year Growth")
            st.plotly_chart(fig_yoy(df,freq), use_container_width=True)
        with cb:
            divider("Descriptive Statistics")
            rows="".join([
                stat_row("Observations",  f"{len(s):,}"),
                stat_row("Mean",          f"{s.mean():,.2f}"),
                stat_row("Median",        f"{s.median():,.2f}"),
                stat_row("Std Deviation", f"{s.std():,.2f}"),
                stat_row("Min / Max",     f"{s.min():,.0f} / {s.max():,.0f}"),
                stat_row("Skewness",      f"{s.skew():.4f}"),
                stat_row("Kurtosis",      f"{s.kurt():.4f}"),
                stat_row("Volatility σ",  f"{vol:.2f}%"),
                stat_row("CAGR",          f"{_cagr:.2f}%"),
                stat_row("Max Drawdown",  f"{_mdd:.2f}%"),
                stat_row("Sharpe Ratio",  f"{_sh:.3f}"),
            ])
            card(rows)

        cc,cd = st.columns([2,3])
        with cc:
            divider("Return Distribution")
            st.plotly_chart(fig_returns(s), use_container_width=True)
        with cd:
            divider(f"Anomaly Flags · {n_an} detected")
            if n_an == 0:
                signal_box("bull","✓","<strong>No anomalies detected.</strong> Data quality check passed.")
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

        skew_txt = ("positive skew — upside tail" if s.skew()>0.5
                    else "negative skew — downside tail" if s.skew()<-0.5
                    else "near-symmetric")
        analyst_note(
            f"Series shows <strong>{skew_txt}</strong>. CAGR <strong>{_cagr:.1f}%</strong>, "
            f"volatility <strong>{vol:.1f}%</strong>, Sharpe <strong>{_sh:.2f}</strong>"
            + (" — strong risk-adjusted return." if _sh>=1
               else " — adequate return." if _sh>=0
               else " — below risk-free rate.") +
            f" Max drawdown <strong>{_mdd:.1f}%</strong>." +
            (f" <strong>{n_an} anomal{'y' if n_an==1 else 'ies'}</strong> detected — consider winsorising." if n_an
             else " Data is clean for modelling."),
            head="Data Intelligence Summary")

    # ══ TAB 2 ══════════════════════════════════════════════════════════
    with t2:
        if not run:
            empty_state("Configure parameters and click  ▶ RUN FORECAST ENGINE",
                        "Select models · Set horizon · Choose scenario")
            return

        fcs, ins_d = {}, {}
        mc_paths = mc_fwd = None

        with st.spinner("Running models — may take 20–40 seconds…"):
            if use_p:
                fwd,ins=run_prophet(df,horizon,freq)
                if fwd is not None: fcs["Prophet"]=fwd; ins_d["Prophet"]=ins
                else: st.warning("Prophet: install with `pip install prophet`")
            if use_a:
                fwd,ins=run_arima(df,horizon,freq)
                if fwd is not None: fcs["ARIMA"]=fwd; ins_d["ARIMA"]=ins
                else: st.warning("ARIMA: install statsmodels")
            if use_x:
                fwd,ins=run_xgboost(df,horizon,freq)
                if fwd is not None: fcs["XGBoost"]=fwd; ins_d["XGBoost"]=ins
                else: st.warning("XGBoost: install xgboost")
            if use_m:
                mc_fwd,mc_paths=run_monte_carlo(df,horizon,freq,n_sims,mc_scen)
                if mc_fwd is not None: fcs["Monte Carlo"]=mc_fwd

        if not fcs: st.error("No models ran. Check installations."); return

        ens = build_ensemble(fcs)
        st.session_state.update({"fcs":fcs,"ins_d":ins_d,"mc_paths":mc_paths,
                                  "mc_fwd":mc_fwd,"ens":ens})

        divider("Multi-Model Forecast")
        st.plotly_chart(fig_forecast(df,fcs,ens), use_container_width=True)

        divider("Terminal Estimates")
        all_models = {**fcs, **({"Ensemble":ens} if ens else {})}
        kpi_bar([(
            name,
            f"{fwd['yhat'].iloc[-1]:,.0f}",
            f"{'▲' if (fwd['yhat'].iloc[-1]-s.iloc[-1])/s.iloc[-1]>=0 else '▼'} "
            f"{abs((fwd['yhat'].iloc[-1]-s.iloc[-1])/s.iloc[-1]*100):.1f}%",
            POS if (fwd['yhat'].iloc[-1]-s.iloc[-1])>=0 else NEG,
            name=="Ensemble"
        ) for name,fwd in all_models.items()])

        divider("Forecast Table")
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
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.session_state["tbl"]=tbl; st.session_state["df_h"]=df

        divider("Forecast Signals")
        if ens is not None:
            ec=(ens["yhat"].iloc[-1]-s.iloc[-1])/s.iloc[-1]*100
            if   ec>5:  signal_box("bull","▲",f"<strong>Bullish:</strong> Consensus projects <strong>+{ec:.1f}%</strong> over {horizon} periods ({scenario}).")
            elif ec<-5: signal_box("bear","▼",f"<strong>Bearish:</strong> Consensus projects <strong>{ec:.1f}%</strong> over {horizon} periods.")
            else:       signal_box("neut","◆",f"<strong>Neutral:</strong> Ensemble projects <strong>{ec:+.1f}%</strong> movement.")
        if n_an>0:
            signal_box("caut","⚠",f"<strong>Data Alert:</strong> {n_an} anomal{'y' if n_an==1 else 'ies'} in training data may widen confidence intervals.")

    # ══ TAB 3 ══════════════════════════════════════════════════════════
    with t3:
        if not run: empty_state("Run the forecast engine first."); return
        mc_paths=st.session_state.get("mc_paths"); mc_fwd=st.session_state.get("mc_fwd")
        if not use_m or mc_paths is None: st.info("Enable Monte Carlo in the sidebar."); return

        divider(f"Monte Carlo Fan · {n_sims:,} Simulations · {scenario}")
        st.plotly_chart(fig_mc_fan(df,mc_paths,mc_fwd["ds"]), use_container_width=True)

        ca,cb = st.columns([3,2])
        with ca:
            divider("Terminal Value Distribution")
            st.plotly_chart(fig_terminal(mc_paths), use_container_width=True)
        with cb:
            tv=mc_paths[:,-1]; var95=np.percentile(tv,5); cvar95=tv[tv<=var95].mean()
            p_loss=np.mean(tv<s.iloc[-1])*100; p_10up=np.mean(tv>s.iloc[-1]*1.1)*100
            divider("Risk Metrics")
            rows="".join([
                stat_row("VaR (95%)",      f"{var95:,.0f}"),
                stat_row("CVaR / ES",      f"{cvar95:,.0f}"),
                stat_row("P(loss)",        f"{p_loss:.1f}%"),
                stat_row("P(gain >10%)",   f"{p_10up:.1f}%"),
                stat_row("Median",         f"{np.percentile(tv,50):,.0f}"),
                stat_row("P10 / P90",      f"{np.percentile(tv,10):,.0f} / {np.percentile(tv,90):,.0f}"),
                stat_row("Scenario",       scenario),
                stat_row("Simulations",    f"{n_sims:,}"),
            ])
            card(rows)
            divider("Percentile Table")
            st.dataframe(pd.DataFrame({
                "Pctl":[f"P{p}" for p in [1,5,10,25,50,75,90,95,99]],
                "Value":[f"{np.percentile(tv,p):,.0f}" for p in [1,5,10,25,50,75,90,95,99]],
                "vs Last":[f"{(np.percentile(tv,p)/s.iloc[-1]-1)*100:+.1f}%" for p in [1,5,10,25,50,75,90,95,99]],
            }), hide_index=True, use_container_width=True)

        analyst_note(
            f"<strong>{n_sims:,} GBM simulations</strong> under <strong>{scenario}</strong>. "
            f"VaR (95%): <strong>{var95:,.0f}</strong> · CVaR: <strong>{cvar95:,.0f}</strong>. "
            f"P(loss): <strong>{p_loss:.1f}%</strong> · P(+10%): <strong>{p_10up:.1f}%</strong>.",
            head="Risk Summary")

    # ══ TAB 4 ══════════════════════════════════════════════════════════
    with t4:
        if not run: empty_state("Run the forecast engine first."); return
        ins_d=st.session_state.get("ins_d",{})
        if not ins_d: st.info("Select at least one parametric model."); return

        scores={}
        for name,ins in ins_d.items():
            merged=df.merge(ins,on="ds",how="inner")
            if merged.empty: continue
            a,f_=merged["y"].values,merged["y_pred"].values
            scores[name]={"MAPE":round(mape(a,f_),3),"sMAPE":round(smape(a,f_),3),"RMSE":round(rmse(a,f_),2)}
        st.session_state["scores"]=scores
        if not scores: st.warning("Could not compute scores."); return

        ranked=sorted(scores,key=lambda m:scores[m]["MAPE"])
        rank_icons=["🥇","🥈","🥉"]

        divider("Model Leaderboard")
        lb_rows=[]
        for i,name in enumerate(ranked):
            sc=scores[name]; col=mc(name)
            conf=max(0,min(100,100-sc["MAPE"]*5))
            bar=f'<div style="display:inline-block;width:{conf:.0f}px;max-width:80px;height:4px;background:{col};border-radius:2px;"></div>'
            best="  ✦ Best Fit" if i==0 else ""
            lb_rows.append({
                "Rank": rank_icons[i] if i<3 else str(i+1),
                "Model": f"{name}{best}",
                "MAPE %": f"{sc['MAPE']:.3f}",
                "sMAPE %": f"{sc['sMAPE']:.3f}",
                "RMSE": f"{sc['RMSE']:,.2f}",
            })
        st.dataframe(pd.DataFrame(lb_rows), hide_index=True, use_container_width=True)

        divider("Accuracy Comparison")
        st.plotly_chart(fig_accuracy(scores), use_container_width=True)

        divider("In-Sample Fit vs Actual")
        st.plotly_chart(fig_fit(df,ins_d), use_container_width=True)

        best=ranked[0]
        gap_txt=""
        if len(ranked)>1:
            gap=abs(scores[ranked[0]]["MAPE"]-scores[ranked[1]]["MAPE"])
            gap_txt=f" Gap to runner-up: <strong>{gap:.2f}pp</strong> — " + \
                    ("ensemble recommended." if gap<1 else "prefer the top model.")
        analyst_note(
            f"<strong>{best}</strong> achieves MAPE <strong>{scores[best]['MAPE']:.2f}%</strong>, "
            f"RMSE <strong>{scores[best]['RMSE']:,.2f}</strong>.{gap_txt} "
            "Next steps: (1) Walk-forward validation on last 20%. "
            "(2) Ensemble top-2 models. (3) Retrain on a rolling window.",
            head="Validation Guidance")

    # ══ TAB 5 ══════════════════════════════════════════════════════════
    with t5:
        divider("Download Outputs")
        ds=datetime.today().strftime("%Y%m%d")
        tbl=st.session_state.get("tbl")
        hist_exp=df.copy(); hist_exp["anomaly_flag"]=mask.astype(int)

        c1,c2,c3 = st.columns(3)
        with c1:
            card(f'<div style="font-weight:700;color:{CR};margin-bottom:0.3rem;">📄 Forecast Table — CSV</div>'
                 f'<div style="font-size:0.75rem;color:{CDM};margin-bottom:0.8rem;">All models with confidence intervals. Excel / Tableau / PowerBI ready.</div>')
            if tbl is not None:
                st.download_button("↓ Download Forecast CSV", to_csv(tbl),
                                   f"FinCast_Forecast_{ds}.csv", "text/csv")
            else: st.caption("Run forecast first.")

        with c2:
            card(f'<div style="font-weight:700;color:{CR};margin-bottom:0.3rem;">📊 Historical Data — CSV</div>'
                 f'<div style="font-size:0.75rem;color:{CDM};margin-bottom:0.8rem;">Cleaned series with anomaly flags. Audit trail ready.</div>')
            st.download_button("↓ Download Historical CSV", to_csv(hist_exp),
                               f"FinCast_Historical_{ds}.csv", "text/csv")

        with c3:
            card(f'<div style="font-weight:700;color:{CR};margin-bottom:0.3rem;">📁 Full Report — Excel</div>'
                 f'<div style="font-size:0.75rem;color:{CDM};margin-bottom:0.8rem;">Multi-sheet: Historical · Forecast · Scores · MC Summary.</div>')
            if tbl is not None:
                sheets={"Historical":hist_exp,"Forecast":tbl}
                sc=st.session_state.get("scores",{})
                if sc:
                    sdf=pd.DataFrame(sc).T.reset_index(); sdf.columns=["Model","MAPE","sMAPE","RMSE"]
                    sheets["Model Accuracy"]=sdf
                mc_fwd2=st.session_state.get("mc_fwd")
                if mc_fwd2 is not None: sheets["MC Summary"]=mc_fwd2[["ds","yhat","lower","upper"]]
                st.download_button("↓ Download Excel Report", to_excel(sheets),
                                   f"FinCast_Report_{ds}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else: st.caption("Run forecast first.")

        if tbl is not None:
            sc=st.session_state.get("scores",{})
            best_m=min(sc,key=lambda m:sc[m]["MAPE"]) if sc else "N/A"
            ens2=st.session_state.get("ens")
            ens_t=f"{ens2['yhat'].iloc[-1]:,.0f}" if ens2 is not None else "N/A"
            analyst_note(
                f"Generated <strong>{datetime.now().strftime('%d %b %Y, %H:%M UTC')}</strong>. "
                f"<strong>{len(df):,} obs</strong> · {freq} · {horizon}-period horizon · {scenario} scenario. "
                f"Best model: <strong>{best_m}</strong>"
                + (f" (MAPE {sc[best_m]['MAPE']:.2f}%)" if best_m!="N/A" else "") +
                f". Ensemble terminal: <strong>{ens_t}</strong>. "
                f"CAGR <strong>{_cagr:.1f}%</strong> · MDD <strong>{_mdd:.1f}%</strong> · Sharpe <strong>{_sh:.2f}</strong>.",
                head="Report Summary")

if __name__ == "__main__":
    main()
