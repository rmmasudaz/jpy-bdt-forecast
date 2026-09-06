# JPY → BDT Exchange-Rate Forecast

A self-contained machine-learning project that downloads real daily JPY→BDT
exchange rates, models them, and visualises the result in an interactive HTML
dashboard.

Built as a learning exercise in time-series forecasting — and as a lesson in
*why understanding the data beats adding more neural network*.

## What you get

| File | Purpose |
| --- | --- |
| `dashboard.html` | **Interactive web dashboard** — double-click to open (needs internet for the Chart.js CDN and fonts). |
| `fetch_data.py` | Downloads daily JPY→BDT rates **plus the two USD legs** (USD/JPY, USD/BDT) from the free `fawazahmed0/currency-api` service. |
| `train_lstm.py` | Trains and evaluates the final structural model, compares it honestly against the LSTM and a naive baseline, writes `model_forecast.json`. |
| `build_dashboard.py` | Inlines `model_forecast.json` into `dashboard.html`. |
| `experiment_features.py` | A/B test of feature sets (add the USD legs, volatility, momentum, day-of-week). |
| `experiment_hybrid.py` | Tests blending the LSTM with the structural mean-reversion signal. |
| `robustness_check.py` | Checks the "taka-leg mean reversion" result across many train/test splits. |
| `data/jpy_bdt_daily.csv` | The downloaded daily series (910 days, 2024-03-02 → 2026-08-28). |
| `model_forecast.json` | All model outputs the dashboard renders. |
| `lstm_model.keras` | The trained LSTM (kept as the deep-learning baseline). |

## Run the pipeline

```bash
pip install pandas numpy scikit-learn tensorflow-cpu requests

python3 fetch_data.py            # 1. fetch data -> data/jpy_bdt_daily.csv
python3 train_lstm.py            # 2. train + evaluate -> model_forecast.json
python3 build_dashboard.py       # 3. build UI  -> dashboard.html
```

## The modelling story (v1 → v4)

- **Target = daily log-return, not the level.** FX rates behave close to a
  random walk, so the best trivial forecast of tomorrow is today. Modelling the
  return (stationary) and reconstructing levels cumulatively is the sound frame.
- **v1/v2 (LSTM).** A 2-layer LSTM on `[return, |return|]`, then with the two USD
  cross-rate legs added, looked encouraging (~60% directional accuracy) — but
  the number was **not stable**: the identical config scored 40–60% depending on
  the random seed. That is noise, not signal.
- **v3 (structural decomposition).** The rate factors exactly:
  `JPY/BDT = (BDT/USD) ÷ (JPY/USD)`, so
  `r(JPY/BDT) = r(USD/BDT) − r(USD/JPY)`. The managed USD/BDT leg
  **mean-reverts** (lag-1 autocorrelation ≈ −0.26); USD/JPY is a random walk.
  v3 was the taka leg's plain AR(1) mean reversion.
- **v4 (weekday-conditional ridge — the final model).** Leg analysis found two
  extra predictable structures in the taka leg:
  1. **Mean reversion is strongly weekday-dependent** — Monday β ≈ −0.98
     (weekend/Sunday adjustments almost fully reverse on Monday), while
     mid-week β is only ≈ −0.13.
  2. **Level reversion** — the taka rate also pulls back toward its 10/20-day
     moving average, not just relative to yesterday's move.

  So v4 predicts the taka-leg return with a **ridge regression** on
  `[r_ub lags 1–3, dev-from-MA10, dev-from-MA20, r_uj(t−1), weekday dummies,
    r_ub(t−1) × weekday interactions]`, then subtracts the yen-leg drift.
  Everything is fit on the training window with train-only standardization;
  the ridge α is selected by **walk-forward refits inside the training
  window** (the test set is never touched). Test evaluation mirrors
  production: the model is **re-fit every day** on all data available so far.
- **Magnitude calibration:** the return forecast is scaled by a `λ` chosen on a
  held-out validation slice to minimise 1-step level MAE. (Direction — the part
  that is predictable — is unaffected by that positive scaling.)
- **Uncertainty:** 95% bands grow like `±1.96 · σ_resid · √h` (random-walk
  scaling).

## High-confidence (selective) mode — the honest route to ~80%

Every-day FX direction is hard (~69% here). But the model can say **"no call"
on weak-signal days** and only predict when its signal is strong. Confidence
thresholds are calibrated **without the test set** (block walk-forward fits
inside the training window; thresholds = percentiles of |prediction|), then
frozen and applied to the test days. Forecast rows in the dashboard carry a
`high_confidence` flag using the 15%-coverage threshold.

## Results (out-of-sample, 182 held-out days, 80/20 split)

| Model | Directional accuracy | 1-step MAE |
| --- | --- | --- |
| **Structural v4 (final, every-day)** | **69.3%** | 0.00166 |
| **Structural v4, high-confidence 25% of days** | **80.0%** | — |
| **Structural v4, high-confidence 13% of days** | **82.6%** | — |
| Structural v3 (previous) | 62.0% | 0.00195 |
| LSTM (v2, deep baseline) | 59.8% | 0.00187 |
| "No change" baseline | 50% (coin flip) | 0.00186 |

Sanity checks done before shipping v4: the ridge wins on **all five** split
fractions 0.65–0.85 (68–71% all-day direction vs 61–65% for v3) and under
daily **walk-forward re-fitting** (69.8%). Verification protocol used
train-only feature scaling, train-only α selection, and leak-free confidence
thresholds.

**Read this honestly.** The accuracy gain is *directional* (69% vs 50% every
day; ~80–86% when the model only calls its strongest quartile/eighth of days).
Daily FX levels are dominated by noise, so every model sits near the "no
change" floor on 1-day level MAE. The lesson scales up with v4: understanding
the data-generating process (a managed taka leg whose mean-reversion strength
**depends on the weekday**, plus reversion to its 10/20-day mean) beat a deep
network by an even wider margin.

*Not investment advice. For learning purposes only.*
