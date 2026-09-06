# JPY → BDT Exchange-Rate Forecast

A self-contained machine-learning project that downloads real daily JPY→BDT
exchange rates, models them, and visualises the result in an interactive HTML
dashboard.

![JPY→BDT Forecast Dashboard](crop_hero.png)

Built as a learning exercise in time-series forecasting — and as a lesson in
*why understanding the data beats adding more neural network*.

## Project Structure & Files

This project follows a modular design with clear separation of concerns. Here's a detailed breakdown:

### **Core Application Files**
| File | Purpose |
| --- | --- |
| `dashboard.html` | **Interactive web dashboard** — double-click to open (needs internet for Chart.js CDN). Features real-time visualization of: <br>- Current and forecasted JPY→BDT rates<br>- Model performance metrics<br>- High-confidence forecast indicators<br>- Historical trends and volatility |
| `fetch_data.py` | Downloads **real daily exchange rate data** from the free `fawazahmed0/currency-api` service. Fetches three currency pairs: <br>- USD/JPY (Yen leg - free floating)<br>- USD/BDT (Taka leg - managed by Bangladesh Bank)<br>- Calculates JPY/BDT as (BDT per USD) / (JPY per USD) |
| `train_lstm.py` | **Main training and evaluation pipeline**: <br>- Trains v4 structural model (weekday-conditional ridge regression)<br>- Trains v2 LSTM model as deep learning baseline<br>- Evaluates on 80/20 split (182-day held-out test set)<br>- Calculates directional accuracy (69.3% for v4)<br>- Performs walk-forward validation<br>- Generates `model_forecast.json` |
| `build_dashboard.py` | Inlines `model_forecast.json` into `dashboard.html` for standalone operation. |

### **Experimental Files**
| File | Purpose |
| --- | --- |
| `experiment_features.py` | A/B testing of feature sets (USD legs, volatility, momentum, day-of-week). |
| `experiment_hybrid.py` | Tests blending LSTM predictions with structural mean-reversion signal. |
| `experiment_phased.py` | Phased accuracy improvement experiments (ridge, LightGBM, ensemble). |
| `robustness_check.py` | Validates v4 model robustness across multiple train/test splits (68-71% accuracy range). |

### **Data & Models**
| File | Purpose | Size |
| --- | --- | --- |
| `data/jpy_bdt_daily.csv` | **Historical exchange rate data** (910 days, 2024-03-02 → 2026-08-28). Contains: <br>- JPY→BDT rate<br>- BDT→JPY reciprocal<br>- USD→JPY (Yen leg)<br>- USD→BDT (Taka leg) | 250KB |
| `model_forecast.json` | **Model outputs and evaluation results**. Contains: <br>- Training/test splits<br>- Forecasted levels and confidence bands<br>- Directional accuracy metrics<br>- High-confidence forecast thresholds<br>- LSTM vs. structural model comparison | 81KB |
| `lstm_model.keras` | **Trained LSTM deep learning model** (2-layer, 64/32 units, dropout 0.2). Serves as baseline for comparison. | 398KB |
| `structural_diagnostics.json` | Diagnostic information about the structural model. | 420B |
| `experiment_results.json` | Results from phased experiment comparisons. | 3.6KB |

### **Documentation & Configuration**
| File | Purpose |
| --- | --- |
| `README.md` | **English project documentation** (you're reading it!) |
| `README_ja.md` | **Japanese translation** of README.md |
| `CONTRIBUTING.md` | Contribution guidelines and development setup |
| `LICENSE` | MIT open source license |
| `requirements.txt` | Python dependencies (pandas, numpy, scikit-learn, tensorflow, requests) |
| `.gitignore` | Git ignore rules for Python/ML projects |
| `.github/CODEOWNERS` | Repository ownership assignments |
| `.github/workflows/` | GitHub Actions CI/CD workflows |

## Run the pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individual packages
pip install pandas numpy scikit-learn tensorflow-cpu requests

# Run the pipeline
python3 fetch_data.py            # 1. fetch data -> data/jpy_bdt_daily.csv
python3 train_lstm.py            # 2. train + evaluate -> model_forecast.json
python3 build_dashboard.py       # 3. build UI  -> dashboard.html
```

## GitHub Notes

- **Git LFS Required:** This project uses Git LFS to track large files like `lstm_model.keras` (~400KB). You may need to install Git LFS before cloning.
- **Japanese Version:** See [README_ja.md](README_ja.md) for Japanese documentation.
- **Repository Protection:** Main branch is protected - all changes require pull request review and passing CI checks.
- **CI/CD:** Automatic tests run on every PR, checking dataset validity and model functionality.

## Model Architecture & Evolution

This project explores **multiple modeling approaches** to forecast JPY→BDT exchange rates, with a focus on understanding the underlying currency market mechanics rather than just using complex neural networks.

### **Key Insight: Currency Pair Decomposition**
The project starts with a crucial observation about currency markets:
```
JPY/BDT = (BDT per USD) / (JPY per USD)
r(JPY/BDT) = r(USD/BDT) − r(USD/JPY)
```

**Why this matters:**
- **USD/JPY (Yen leg):** Free-floating market rate (random walk behavior)
- **USD/BDT (Taka leg):** Managed rate by Bangladesh Bank (exhibits strong mean-reversion)

### **Model Versions**

#### **v1: Simple LSTM (Baseline)**
- **Architecture:** 2-layer LSTM with 64/32 units + dropout (0.2)
- **Features:** Daily log returns
- **Performance:** ~60% directional accuracy (unstable - 40-60% range depending on random seed)
- **Limitation:** Lacks understanding of currency market structure

#### **v2: Enhanced LSTM**
- **Improvement:** Added USD cross-rate legs as features
- **Features:** [return, |return|, USD/BDT, USD/JPY]
- **Performance:** ~60% directional accuracy (still unstable)
- **Limitation:** Complex model without structural insight

#### **v3: Structural AR(1) Model**
- **Breakthrough:** Focus on currency decomposition
- **Approach:** Predict USD/BDT mean-reversion, leave USD/JPY as random walk
- **Features:** USD/BDT lagged returns
- **Performance:** 62.0% directional accuracy (stable across splits)
- **Key Finding:** USD/BDT has lag-1 autocorrelation of -0.26 (strong mean-reversion)

#### **v4: Weekday-Conditional Ridge Regression (Final Model)**
- **Architecture:** Ridge regression with weekday-dependent features
- **Features:**
  - USD/BDT lagged returns (1-3 days)
  - Deviation from 10/20-day moving averages (level reversion)
  - USD/JPY lagged returns (random walk baseline)
  - Weekday dummies (Monday-Friday indicators)
  - Interaction terms: USD/BDT lag × weekday (captures day-of-week effects)
- **Key Innovation:** 
  - Monday effect: β ≈ -0.98 (weekend adjustments reverse almost completely)
  - Mid-week effect: β ≈ -0.13 (weak mean-reversion)
- **Performance:** **69.3% directional accuracy** (stable across all splits)
- **High-Confidence Mode:** 80.0% accuracy when only calling strongest 25% of signals

### **Model Evaluation Protocol**
- **80/20 chronological split:** 728 days training, 182 days test
- **Walk-forward validation:** Model re-fit daily on available data
- **Train-only calibration:** All hyperparameters selected without touching test data
- **No data leakage:** Strict separation between training, validation, and test sets

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
