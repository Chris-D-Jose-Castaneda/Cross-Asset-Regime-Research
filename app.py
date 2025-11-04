# app.py — Macro-Credit Regime Radar (dynamic, concise)

from __future__ import annotations
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st, yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import roc_auc_score
from matplotlib.ticker import PercentFormatter
from dotenv import load_dotenv
from fredapi import Fred

# ── Env / keys ────────────────────────────────────────────────────────────────
# Put FRED_API_KEY=your_key in .env at repo root
load_dotenv()
API_KEY = os.getenv("FRED_API_KEY", "")
USE_FRED = bool(API_KEY)

# ── Page / style ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Macro-Credit Regime Radar", layout="wide")
plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True})

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("Settings")
START       = st.sidebar.date_input("Start", pd.Timestamp("2002-01-01")).isoformat()
ROLL_YEARS  = st.sidebar.slider("Composite rolling z window (years)", 2, 10, 5)
SMOOTH_W    = st.sidebar.slider("Composite smoothing (weeks)", 1, 12, 4)
K_STATES    = st.sidebar.selectbox("Regime states", [2, 3], index=0)
FREQ        = "W-FRI"

DEFAULT_WEIGHTS = {
    "PCE_yoy": -1.0, "RetailSales_yoy": -0.8, "UnemploymentRate": 1.0,
    "Claims_yoy": 0.8, "TED_spread": 1.0, "HY_OAS": 1.2, "IG_OAS": 0.8,
}
st.sidebar.subheader("Composite weights (↑ = more stress)")
WEIGHTS = {k: st.sidebar.number_input(k, value=float(v), step=0.1, format="%.1f")
           for k, v in DEFAULT_WEIGHTS.items()}

st.sidebar.subheader("Betas")
BETA_YEARS  = st.sidebar.slider("Beta lookback (years)", 1, 10, 5)
BETA_REGIME = st.sidebar.selectbox("Condition betas on", ["All data", "Low-stress only", "High-stress only"], index=0)

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

# ── FRED series map ────────────────────────────────────────────────────────────
FRED_SERIES = {
    "PCE_yoy"         : ("PCE", "pc1"),
    "RetailSales_yoy" : ("RSAFS", "pc1"),
    "UnemploymentRate": ("UNRATE", None),
    "Claims"          : ("ICSA", None),
    "TED_spread"      : ("TEDRATE", None),
    "HY_OAS"          : ("BAMLH0A0HYM2", None),
    "IG_OAS"          : ("BAMLC0A0CM", None),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def weekly_align(df: pd.DataFrame, freq=FREQ) -> pd.DataFrame:
    return df.resample(freq).last().ffill()

def zscore_rolling(df: pd.DataFrame, years: int) -> pd.DataFrame:
    win = years * 52
    mu  = df.rolling(win, min_periods=win//2).mean()
    sd  = df.rolling(win, min_periods=win//2).std(ddof=0)
    return (df - mu) / sd

@st.cache_data(ttl=1800)
def load_macro(start: str) -> pd.DataFrame:
    if not USE_FRED:
        return pd.DataFrame()
    fred = Fred(api_key=API_KEY)
    cols = {}
    for key, (sid, units) in FRED_SERIES.items():
        s = fred.get_series(sid, units=units) if units else fred.get_series(sid)
        s.name = key
        cols[key] = s
    df = pd.concat(cols.values(), axis=1).sort_index()
    df = df.loc[df.index >= start]
    df = weekly_align(df)
    df["Claims_yoy"] = df["Claims"].pct_change(52) * 100.0
    keep = [k for k in WEIGHTS if k in df.columns]
    return df[keep]

@st.cache_data(ttl=1800)
def load_markets(start: str) -> pd.DataFrame:
    tickers = ["SPY", "HYG", "LQD"]
    px = yf.download(tickers, period="max", auto_adjust=True, progress=False)["Close"]
    px = px.loc[px.index >= start]
    pxw = weekly_align(px)
    rets = pxw.pct_change().rename(columns=lambda c: f"{c}_ret")
    tr   = (1 + rets).cumprod() - 1
    tr.columns = [c.replace("_ret", "_cum") for c in tr.columns]
    return pxw.join(rets).join(tr)

def build_mcsr(macro: pd.DataFrame, weights: dict, roll_years: int, smooth_w: int):
    drivers = [k for k in weights if k in macro.columns]
    Z = zscore_rolling(macro[drivers], roll_years).add_suffix("_z")
    w = pd.Series({f"{k}_z": weights[k] for k in drivers})
    mcsr = (Z * w).sum(axis=1).to_frame("MCSR")
    mcsr["MCSR_smooth"] = mcsr["MCSR"].rolling(smooth_w, min_periods=1).mean()
    return mcsr, drivers

def fit_regimes(series: pd.Series, k: int):
    y = (series - series.mean()) / series.std(ddof=0)
    y = y.dropna()
    mr = MarkovRegression(y.values, k_regimes=k, trend="c", switching_variance=True)
    fit = mr.fit(disp=False, maxiter=200, em_iter=20)
    probs = np.array(fit.smoothed_marginal_probabilities)
    if probs.shape[0] <= probs.shape[1]:
        probs = probs.T
    P = pd.DataFrame(probs, index=y.index, columns=[f"Regime{i}" for i in range(k)])
    means = []
    for j in range(k):
        pj = P.iloc[:, j].values
        means.append((y.values * pj).sum() / pj.sum())
    hi = int(np.argmax(means))
    Phigh  = P.iloc[:, hi].rename("P(HighStress)")
    Regime = (Phigh > 0.5).astype(int).rename("Regime")
    return Phigh, Regime, dict(zip(P.columns, np.round(means, 3))), hi

def shade(ax, mask: pd.Series):
    on = mask.astype(bool)
    blocks = (on != on.shift()).cumsum()
    for _, seg in on[on].groupby(blocks[on]):
        ax.axvspan(seg.index[0], seg.index[-1], color="grey", alpha=0.2)

def ridge_betas(X: pd.DataFrame, Y: pd.DataFrame, ridge=1e-6) -> pd.DataFrame:
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    XtY = X.T @ Y
    B = np.linalg.solve(XtX, XtY)
    return pd.DataFrame(B, index=X.columns, columns=Y.columns)

# ── Load data ──────────────────────────────────────────────────────────────────
macro = load_macro(START)
mkts  = load_markets(START)

st.title("Macro-Credit Regime Radar")

if macro.empty:
    st.warning("No macro data loaded. Ensure FRED_API_KEY is set in your .env file. Showing SPY only.")
    fig, ax = plt.subplots()
    mkts["SPY_cum"].plot(ax=ax, label="SPY total return")
    ax.legend()
    st.pyplot(fig)
    st.stop()

# Composite + regimes
mcsr, drivers = build_mcsr(macro, WEIGHTS, ROLL_YEARS, SMOOTH_W)
Phigh, Regime, means, hi_idx = fit_regimes(mcsr["MCSR_smooth"], K_STATES)
data = macro.join(mcsr).join(mkts)

# ── Top: composite + probability ───────────────────────────────────────────────
c1, c2 = st.columns([2,1])
with c1:
    st.subheader("Composite stress index (MCSR)")
    fig, ax = plt.subplots()
    mcsr["MCSR"].plot(ax=ax, label="MCSR")
    mcsr["MCSR_smooth"].plot(ax=ax, label="MCSR (smooth)")
    ax.set_ylabel("z-weighted level")
    ax.legend()
    st.pyplot(fig)
with c2:
    st.subheader("P(High-stress)")
    fig, ax = plt.subplots()
    Phigh.plot(ax=ax)
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    st.caption(f"Regime means (std): {means}.  High-stress = Regime{hi_idx}")

# ── Overlay: shaded regimes on returns ─────────────────────────────────────────
st.subheader("Regimes over total return (shaded = high stress)")
picked = st.multiselect("Series", ["SPY_cum","HYG_cum","LQD_cum"], default=["SPY_cum","HYG_cum","LQD_cum"])
overlay = data[picked].join(Regime).dropna()
fig, ax = plt.subplots(figsize=(10,5))
overlay[picked].plot(ax=ax)
shade(ax, overlay["Regime"]==1)
ax.set_title("Total return with high-stress shading")
st.pyplot(fig)

# ── Dynamic betas (window + regime-conditioned) ────────────────────────────────
st.subheader("Driver → Asset betas (bp per 1σ driver shock, weekly)")
X_all = data[drivers].pct_change().replace([np.inf, -np.inf], np.nan)
X_all = (X_all - X_all.mean()) / X_all.std(ddof=0)
Y_all = data[["SPY_ret","HYG_ret","LQD_ret"]]

cutoff = X_all.index.max() - pd.Timedelta(weeks=BETA_YEARS*52)
Xw, Yw = X_all.loc[X_all.index >= cutoff], Y_all.loc[Y_all.index >= cutoff]

mask = pd.Series(True, index=Xw.index)
if BETA_REGIME != "All data":
    reg = (Phigh > 0.5).astype(int).reindex(Xw.index).ffill().bfill()
    mask = reg.eq(1 if BETA_REGIME.startswith("High") else 0)
Xw, Yw = Xw[mask], Yw[mask]
XY = Xw.join(Yw).dropna()

betas = ridge_betas(XY[drivers], XY[["SPY_ret","HYG_ret","LQD_ret"]])
st.dataframe((betas * 10000).round(2))
st.caption(f"Window: last {BETA_YEARS}y • Regime: {BETA_REGIME} • Obs: {len(XY):,}")

# ── Scenario engine (use chosen betas: all-data or regime-conditioned) ─────────
st.subheader("Scenario engine")
cols = st.columns(3)
with cols[0]:
    z_hy  = st.slider("HY_OAS (σ)",  -3.0, 3.0,  1.5, 0.1)
    z_ig  = st.slider("IG_OAS (σ)",  -3.0, 3.0,  0.8, 0.1)
with cols[1]:
    z_ted = st.slider("TED_spread (σ)", -3.0, 3.0, 1.0, 0.1)
    z_ur  = st.slider("UnemploymentRate (σ)", -3.0, 3.0, 0.5, 0.1)
with cols[2]:
    z_pce = st.slider("PCE_yoy (σ)", -3.0, 3.0, -0.5, 0.1)
    z_rsa = st.slider("RetailSales_yoy (σ)", -3.0, 3.0, -0.5, 0.1)

zvec = pd.Series({
    "HY_OAS": z_hy, "IG_OAS": z_ig, "TED_spread": z_ted,
    "UnemploymentRate": z_ur, "PCE_yoy": z_pce, "RetailSales_yoy": z_rsa
}).reindex(drivers).fillna(0.0)

proj = (betas.T @ zvec).sort_values()
mcsr_dz = pd.Series(WEIGHTS).reindex(drivers).fillna(0.0) @ zvec

c3, c4 = st.columns([2,1])
with c3:
    fig, ax = plt.subplots()
    (proj * 100).plot(kind="barh", ax=ax, title="Projected 1-week returns (%)")
    ax.set_xlabel("Projected return (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    xmax = float(np.nanmax(np.abs(proj))) * 100
    if xmax < 0.5:
        ax.set_xlim(-0.6, 0.6)
    st.pyplot(fig)
with c4:
    st.metric("Implied ΔMCSR (z-units)", f"{mcsr_dz:+.2f}")

# ── By-regime performance & OOS AUC ───────────────────────────────────────────
st.subheader("By-regime performance (weekly, %)")
tmp = data[["SPY_ret","HYG_ret","LQD_ret"]].join((Phigh>0.5).astype(int).rename("Regime")).dropna()
st.dataframe((tmp.groupby("Regime")[["SPY_ret","HYG_ret","LQD_ret"]].agg(["mean","std"])*100).round(2))

st.subheader("Out-of-sample AUC (stress proxy)")
oos_cut = int(len(Phigh)*0.7)
oos_idx = Phigh.index[oos_cut:]
thr = data.reindex(oos_idx)["HY_OAS"].quantile(0.8)
y_true = (data.reindex(oos_idx)["HY_OAS"] >= thr).astype(int)
auc = roc_auc_score(y_true, Phigh.reindex(oos_idx).values)
st.write(f"AUC vs HY_OAS≥80th (OOS): **{auc:.3f}**")

# ── Export ────────────────────────────────────────────────────────────────────
export_df = mcsr.join(Phigh).join(Regime)
st.download_button("⬇️ Download composite & regimes (CSV)",
                   export_df.to_csv().encode(), "mcsr_regimes.csv", "text/csv")

st.caption("Dynamic build: rolling-z composite, 2–3 state regimes, shaded overlays, window/regime-conditioned betas, and scenario P/L.")
