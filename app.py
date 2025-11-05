# app.py — Macro-Credit Regime Radar (robust, no-error)

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from sklearn.metrics import roc_auc_score
from matplotlib.ticker import PercentFormatter
from dotenv import load_dotenv

# Optional import: fredapi (only used if key present)
try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except Exception:
    _FRED_AVAILABLE = False

# ── Env / keys ────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("FRED_API_KEY", "") or os.getenv("FRED_API_KEY".lower(), "")
USE_FRED = bool(API_KEY and _FRED_AVAILABLE)

# ── Page / style ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Macro-Credit Regime Radar", layout="wide")
plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True})

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Settings")
START       = st.sidebar.date_input("Start", pd.Timestamp("2002-01-01")).isoformat()
ROLL_YEARS  = st.sidebar.slider("Composite rolling z window (years)", 2, 10, 5)
SMOOTH_W    = st.sidebar.slider("Composite smoothing (weeks)", 1, 12, 4)
K_STATES    = st.sidebar.selectbox("Regime states", [2, 3], index=0)
FREQ        = "W-FRI"

DEFAULT_WEIGHTS = {
    "PCE_yoy": -1.0,
    "RetailSales_yoy": -0.8,
    "UnemploymentRate": 1.0,
    "Claims_yoy": 0.8,
    "TED_spread": 1.0,
    "HY_OAS": 1.2,
    "IG_OAS": 0.8,
}
st.sidebar.subheader("Composite weights (↑ = more stress)")
WEIGHTS = {k: st.sidebar.number_input(k, value=float(v), step=0.1, format="%.1f")
           for k, v in DEFAULT_WEIGHTS.items()}

st.sidebar.subheader("Betas")
BETA_YEARS  = st.sidebar.slider("Beta lookback (years)", 1, 10, 5)
BETA_REGIME = st.sidebar.selectbox("Condition betas on",
                                   ["All data", "Low-stress only", "High-stress only"], index=0)

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()

# ── FRED series map ───────────────────────────────────────────────────────────
FRED_SERIES = {
    "PCE_yoy"         : ("PCE", "pc1"),        # YoY %
    "RetailSales_yoy" : ("RSAFS", "pc1"),      # YoY %
    "UnemploymentRate": ("UNRATE", None),      # %
    "Claims"          : ("ICSA", None),        # level (initial claims)
    "TED_spread"      : ("TEDRATE", None),     # %
    "HY_OAS"          : ("BAMLH0A0HYM2", None),# %
    "IG_OAS"          : ("BAMLC0A0CM", None),  # %
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_datetime_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    obj = df.copy()
    if not isinstance(obj.index, pd.DatetimeIndex):
        obj.index = pd.to_datetime(obj.index)
    return obj.sort_index()

def weekly_align(df: pd.DataFrame, freq="W-FRI") -> pd.DataFrame:
    df = ensure_datetime_index(df)
    return df.resample(freq).last().ffill()

def zscore_rolling(df: pd.DataFrame, years: int) -> pd.DataFrame:
    df = df.copy()
    win = max(int(years * 52), 4)
    mu  = df.rolling(win, min_periods=win//2).mean()
    sd  = df.rolling(win, min_periods=win//2).std(ddof=0)
    out = (df - mu) / sd
    return out

@st.cache_data(ttl=1800)
def load_macro(start: str) -> pd.DataFrame:
    if not USE_FRED:
        return pd.DataFrame()
    fred = Fred(api_key=API_KEY)
    cols: dict[str, pd.Series] = {}
    for key, (sid, units) in FRED_SERIES.items():
        if units:
            s = fred.get_series(sid, units=units)
        else:
            s = fred.get_series(sid)
        s.name = key
        cols[key] = s
    df = pd.concat(cols.values(), axis=1)
    df = df.loc[df.index >= pd.to_datetime(start)]
    df = weekly_align(df)
    if "Claims" in df.columns:
        df["Claims_yoy"] = df["Claims"].pct_change(52) * 100.0
    keep = [k for k in WEIGHTS if k in df.columns]
    return df[keep].dropna(how="all")

@st.cache_data(ttl=1800)
def load_markets(start: str) -> pd.DataFrame:
    tickers = ["SPY", "HYG", "LQD"]
    px = yf.download(tickers, period="max", auto_adjust=True, progress=False)["Close"]
    px = px.loc[px.index >= pd.to_datetime(start)]
    pxw = weekly_align(px)
    rets = pxw.pct_change()
    rets.columns = [f"{c}_ret" for c in rets.columns]
    tr   = (1.0 + rets).cumprod() - 1.0
    tr.columns = [c.replace("_ret", "_cum") for c in rets.columns]
    out = pxw.join(rets, how="left").join(tr, how="left")
    return out

def build_mcsr(macro: pd.DataFrame, weights: dict, roll_years: int, smooth_w: int):
    drivers = [k for k in weights if k in macro.columns]
    if not drivers:
        return pd.DataFrame(columns=["MCSR", "MCSR_smooth"]), []
    Z = zscore_rolling(macro[drivers], roll_years).add_suffix("_z")
    w = pd.Series({f"{k}_z": float(weights[k]) for k in drivers})
    mcsr = (Z * w).sum(axis=1).to_frame("MCSR")
    mcsr["MCSR_smooth"] = mcsr["MCSR"].rolling(max(int(smooth_w), 1), min_periods=1).mean()
    return mcsr.dropna(how="all"), drivers

def fit_regimes(series: pd.Series, k: int):
    s = series.dropna()
    if len(s) < max(60, 10 * k):
        # Not enough data to fit a Markov model; return a simple z-rule proxy
        z = (s - s.mean()) / (s.std(ddof=0) or 1.0)
        Phigh = (z - z.min()) / ((z.max() - z.min()) or 1.0)
        Regime = (Phigh > 0.5).astype(int).rename("Regime")
        means = {"proxy_low": float(z[z <= 0].mean() or 0.0),
                 "proxy_high": float(z[z > 0].mean() or 0.0)}
        hi_idx = int(1)
        return Phigh.rename("P(HighStress)"), Regime, means, hi_idx

    y = (s - s.mean()) / (s.std(ddof=0) or 1.0)
    mr = MarkovRegression(y.values, k_regimes=int(k), trend="c", switching_variance=True)
    fit = mr.fit(disp=False, maxiter=200, em_iter=20)

    probs = np.array(fit.smoothed_marginal_probabilities)
    if probs.shape[0] <= probs.shape[1]:
        probs = probs.T  # shape: (T, k)
    P = pd.DataFrame(probs, index=y.index, columns=[f"Regime{i}" for i in range(int(k))])

    means = []
    for j in range(int(k)):
        pj = P.iloc[:, j].values
        denom = float(pj.sum() or np.nan)
        m = float((y.values * pj).sum() / denom) if np.isfinite(denom) else 0.0
        means.append(m)
    hi = int(np.nanargmax(means))
    Phigh = P.iloc[:, hi].rename("P(HighStress)").clip(0, 1)
    Regime = (Phigh > 0.5).astype(int).rename("Regime")
    return Phigh, Regime, dict(zip(P.columns, np.round(means, 3))), hi

def shade(ax, mask: pd.Series):
    on = mask.astype(bool)
    if on.empty or not on.any():
        return
    blocks = (on != on.shift()).cumsum()
    for _, seg in on[on].groupby(blocks[on]):
        ax.axvspan(seg.index[0], seg.index[-1], color="grey", alpha=0.2)

def ridge_betas(X: pd.DataFrame, Y: pd.DataFrame, ridge=1e-6) -> pd.DataFrame:
    # Solve (X'X + λI)B = X'Y
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    XtY = X.T @ Y
    B = np.linalg.solve(XtX, XtY)
    return pd.DataFrame(B, index=X.columns, columns=Y.columns)

# ── Load data ─────────────────────────────────────────────────────────────────
macro = load_macro(START)
mkts  = load_markets(START)

st.title("Macro-Credit Regime Radar")

if macro.empty:
    st.warning("No macro data loaded. Ensure FRED_API_KEY is set in your .env file (and fredapi installed). Showing SPY only.")
    if mkts.empty or "SPY_cum" not in mkts.columns:
        st.info("Market data unavailable.")
    else:
        fig, ax = plt.subplots()
        mkts["SPY_cum"].dropna().plot(ax=ax, label="SPY total return")
        ax.legend()
        st.pyplot(fig)
    st.stop()

# Composite + regimes
mcsr, drivers = build_mcsr(macro, WEIGHTS, ROLL_YEARS, SMOOTH_W)
if mcsr.empty:
    st.error("Composite could not be constructed (no valid drivers). Adjust weights or FRED start date.")
    st.stop()

Phigh, Regime, means, hi_idx = fit_regimes(mcsr["MCSR_smooth"], K_STATES)

# Align everything to a common weekly index
idx = mcsr.index
if not mkts.empty:
    idx = idx.intersection(mkts.index)
idx = idx.intersection(Phigh.index)

macro = macro.reindex(idx).ffill()
mcsr  = mcsr.reindex(idx).ffill()
Phigh = Phigh.reindex(idx).ffill()
Regime = Regime.reindex(idx).ffill().astype(int)
mkts  = mkts.reindex(idx)

data = macro.join(mcsr, how="left").join(mkts, how="left").join(Phigh, how="left").join(Regime, how="left")
data = data.dropna(how="all")

# ── Top: composite + probability ──────────────────────────────────────────────
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Composite stress index (MCSR)")
    fig, ax = plt.subplots()
    mcsr["MCSR"].dropna().plot(ax=ax, label="MCSR")
    mcsr["MCSR_smooth"].dropna().plot(ax=ax, label="MCSR (smooth)")
    ax.set_ylabel("z-weighted level")
    ax.legend()
    st.pyplot(fig)
with c2:
    st.subheader("P(High-stress)")
    fig, ax = plt.subplots()
    Phigh.dropna().plot(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
    st.caption(f"Regime means (std): {means}.  High-stress = Regime{hi_idx}")

# ── Overlay: shaded regimes on returns ────────────────────────────────────────
st.subheader("Regimes over total return (shaded = high stress)")
available_cum = [c for c in ["SPY_cum", "HYG_cum", "LQD_cum"] if c in data.columns]
picked = st.multiselect("Series", available_cum, default=available_cum)
if picked:
    overlay = data[picked + ["Regime"]].copy()
    overlay = overlay.dropna(subset=picked, how="all")
    if overlay.empty:
        st.info("No overlapping return data to plot for the chosen start date and filters.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        overlay[picked].fillna(method="ffill").plot(ax=ax)
        shade(ax, overlay["Regime"] == 1)
        ax.set_title("Total return with high-stress shading")
        st.pyplot(fig)
else:
    st.info("Select at least one series to plot.")

# ── Dynamic betas (window + regime-conditioned) ───────────────────────────────
st.subheader("Driver → Asset betas (bp per 1σ driver shock, weekly)")
assets = [c for c in ["SPY_ret", "HYG_ret", "LQD_ret"] if c in data.columns]
if drivers and assets:
    X_all = data[drivers].pct_change().replace([np.inf, -np.inf], np.nan)
    X_all = (X_all - X_all.mean()) / (X_all.std(ddof=0).replace(0, np.nan))
    Y_all = data[assets]

    cutoff = X_all.index.max() - pd.Timedelta(weeks=BETA_YEARS * 52)
    Xw, Yw = X_all.loc[X_all.index >= cutoff], Y_all.loc[Y_all.index >= cutoff]

    mask = pd.Series(True, index=Xw.index)
    if BETA_REGIME != "All data":
        reg = (Phigh > 0.5).astype(int).reindex(Xw.index).ffill().bfill()
        mask = reg.eq(1 if BETA_REGIME.startswith("High") else 0)

    XY = Xw[mask].join(Yw[mask], how="inner").dropna()
    if len(XY) >= len(drivers) + 2:
        betas = ridge_betas(XY[drivers], XY[assets])
        st.dataframe((betas * 10000).round(2))
        st.caption(f"Window: last {BETA_YEARS}y • Regime: {BETA_REGIME} • Obs: {len(XY):,}")
    else:
        st.info("Not enough observations in the selected window/regime to estimate betas.")
        betas = pd.DataFrame(columns=assets, index=drivers)
else:
    st.info("Drivers or assets missing; cannot compute betas.")
    betas = pd.DataFrame()

# ── Scenario engine (uses betas if available) ─────────────────────────────────
st.subheader("Scenario engine")
driver_knobs = ["HY_OAS", "IG_OAS", "TED_spread", "UnemploymentRate", "PCE_yoy", "RetailSales_yoy"]
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

if not betas.empty:
    zvec = pd.Series({
        "HY_OAS": z_hy, "IG_OAS": z_ig, "TED_spread": z_ted,
        "UnemploymentRate": z_ur, "PCE_yoy": z_pce, "RetailSales_yoy": z_rsa
    }).reindex(betas.index).fillna(0.0)

    proj = (betas.T @ zvec).sort_values()
    mcsr_dz = pd.Series(WEIGHTS).reindex(zvec.index).fillna(0.0) @ zvec

    c3, c4 = st.columns([2, 1])
    with c3:
        fig, ax = plt.subplots()
        (proj * 100).plot(kind="barh", ax=ax, title="Projected 1-week returns (%)")
        ax.set_xlabel("Projected return (%)")
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        xmax = float(np.nanmax(np.abs(proj))) * 100 if len(proj) else 0.0
        if xmax < 0.5:
            ax.set_xlim(-0.6, 0.6)
        st.pyplot(fig)
    with c4:
        st.metric("Implied ΔMCSR (z-units)", f"{float(mcsr_dz):+0.2f}")
else:
    st.info("Betas unavailable; scenario engine will activate once betas are estimated.")

# ── By-regime performance & OOS AUC ───────────────────────────────────────────
st.subheader("By-regime performance (weekly, %)")
assets_meanstd = [c for c in ["SPY_ret", "HYG_ret", "LQD_ret"] if c in data.columns]
tmp = data[assets_meanstd + ["Regime"]].dropna()
if not tmp.empty:
    st.dataframe((tmp.groupby("Regime")[assets_meanstd].agg(["mean", "std"]) * 100).round(2))
else:
    st.info("Not enough data to compute by-regime performance.")

st.subheader("Out-of-sample AUC (stress proxy)")
if "HY_OAS" in macro.columns and not Phigh.dropna().empty:
    oos_cut = int(len(Phigh.dropna()) * 0.7)
    oos_idx = Phigh.dropna().index[oos_cut:]
    if len(oos_idx) >= 10:
        thr = macro.reindex(oos_idx)["HY_OAS"].quantile(0.8)
        y_true = (macro.reindex(oos_idx)["HY_OAS"] >= thr).astype(int)
        auc = roc_auc_score(y_true, Phigh.reindex(oos_idx).values)
        st.write(f"AUC vs HY_OAS≥80th (OOS): **{auc:.3f}**")
    else:
        st.info("Not enough OOS observations to compute AUC.")
else:
    st.info("HY_OAS or regime probabilities missing; cannot compute AUC.")

# ── Export ────────────────────────────────────────────────────────────────────
export_df = mcsr.join(Phigh).join(Regime)
if not export_df.dropna(how="all").empty:
    st.download_button("⬇️ Download composite & regimes (CSV)",
                       export_df.to_csv().encode(), "mcsr_regimes.csv", "text/csv")

st.caption("Dynamic build: rolling-z composite, 2–3 state regimes, shaded overlays, window/regime-conditioned betas, and scenario P/L.")
