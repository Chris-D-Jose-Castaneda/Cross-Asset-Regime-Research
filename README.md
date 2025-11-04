# Cross Asset Regime Research

A compact repo with two research notebooks and one Streamlit app (works locally but for some reason can't deploy) that builds a weekly macro credit stress signal, learns regimes, and links shocks to asset returns.

## Contents

* **Macro Credit Regime Radar** `app.py`
  Pulls key FRED series and weekly market data, builds a rolling z scored composite stress index, fits a two or three state Markov switching model with high stress shading, estimates regime conditioned betas for SPY HYG LQD, includes a simple scenario engine and CSV export.
* **USD Regimes Metals and FX** notebook
  Equal weighted USD proxy from major FX, metals basket, rolling z scores, PCA, correlations, and a simple metals long regime test. Requires LSEG Workspace CodeBook with the Refinitiv data environment to fully reproduce.
* **Macro Credit Stress Radar** notebook
  Exploratory version of the Streamlit workflow with extra diagnostics.

## Quick start

1. Create and activate a virtual environment, then install deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Add secrets in a `.env` file at the repo root

```env
FRED_API_KEY=your_fred_key
```

3. Run the app

```bash
streamlit run app.py
```

## Notes

* If no FRED key is set the app falls back to an SPY total return view only.
* The USD Regimes Metals and FX notebook uses LSEG code and will not fully run outside the CodeBook environment.
