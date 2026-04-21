# Intraday Stock Price Prediction: Multi-Architecture Comparative Study

## Overview

Compare three deep learning architectures (LSTM, TFT, CNN-LSTM) on predicting intraday stock price direction (up/down/neutral) using OHLCV data + technical indicators for 30 S&P 500 equities.

**Course**: AAI 6640 — Applied Deep Learning
**Framework**: PyTorch 2.x
**Tracking**: MLflow
**Linting**: Ruff

---

## Project Structure

```
AAI-6640-DL-Final-Project/
├── config/
│   └── default.yaml                # All hyperparameters and settings
├── data/
│   └── raw/                        # Cached parquet files (gitignored)
├── src/
│   ├── data/
│   │   ├── download.py             # yfinance download + caching
│   │   ├── features.py             # Stationary OHLCV + 18 indicators + Fourier time features
│   │   ├── labels.py               # Volatility-normalized adaptive label generation
│   │   └── dataset.py              # PyTorch Dataset, windowing, walk-forward splits
│   ├── models/
│   │   ├── lstm.py                 # Stacked LSTM + temporal attention
│   │   ├── tft.py                  # Temporal Fusion Transformer + static covariate encoder
│   │   └── cnn_lstm.py             # Dilated CNN-LSTM Hybrid
│   ├── training/
│   │   ├── trainer.py              # Training loop, focal loss, early stopping, checkpoints
│   │   ├── losses.py               # Focal loss implementation
│   │   └── metrics.py              # Accuracy, F1, confusion matrix, McNemar's test
│   ├── evaluation/
│   │   ├── backtest.py             # Portfolio sim, Sharpe/Sortino/Calmar ratios
│   │   ├── feature_importance.py   # TFT VSN + mutual information + LSTM ablation
│   │   └── volatility_analysis.py  # High-vol vs low-vol breakdown
│   └── utils/
│       └── config.py               # YAML config loader
├── frontend/
│   └── src/                        # Next.js dashboard source
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── pyproject.toml                  # Project metadata + ruff config
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 1. Data Pipeline

### 1.1 Tickers (30 stocks, 6 per sector)

| Sector                 | Tickers                             |
| ---------------------- | ----------------------------------- |
| Technology             | AAPL, MSFT, NVDA, GOOGL, META, AMZN |
| Financials             | JPM, BAC, GS, MS, WFC, C            |
| Healthcare             | JNJ, UNH, PFE, ABT, MRK, TMO        |
| Energy                 | XOM, CVX, COP, SLB, EOG, MPC        |
| Consumer Discretionary | TSLA, HD, NKE, MCD, SBUX, TJX       |

### 1.2 Data Source

- **Provider**: **yfinance** (open data, no API key required)
- **Data source policy**: single provider (yfinance)
- **Intervals**: 1-minute, 5-minute, 15-minute
- **History**: 252 trading days (~1 year) of 5-minute bars per ticker
- **Storage**: Cache as parquet files in `data/raw/`
- **Auth**: Not required for default setup

### 1.3 Features (5 stationary OHLCV + 18 indicators + 5 temporal = 28 total)

Raw OHLCV prices are non-stationary (price levels drift over time). We replace them with stationary return-based representations.

**Stationary OHLCV replacements (5):**

| #   | Feature           | Formula                            | Rationale                                |
| --- | ----------------- | ---------------------------------- | ---------------------------------------- |
| 1   | Log return        | `ln(Close_t / Close_{t-1})`        | Additive, ~normal, price-level invariant |
| 2   | High-Low range    | `(High - Low) / Close`             | Intrabar volatility as fraction of price |
| 3   | Open-Close body   | `(Close - Open) / Close`           | Intrabar directional move                |
| 4   | Upper shadow      | `(High - max(Open,Close)) / Close` | Buying pressure rejection                |
| 5   | Volume log-change | `ln(Vol_t / Vol_{t-1})`            | Stationary volume representation         |

**Technical indicators (18) — via `pandas_ta`:**

| #   | Feature         | Parameters              | Category            |
| --- | --------------- | ----------------------- | ------------------- |
| 6   | RSI             | period=14               | Momentum            |
| 7   | MACD Line       | fast=12, slow=26, sig=9 | Trend               |
| 8   | MACD Signal     | (same)                  | Trend               |
| 9   | MACD Histogram  | (same)                  | Trend strength      |
| 10  | Bollinger Upper | period=20, std=2        | Volatility band     |
| 11  | Bollinger Lower | (same)                  | Volatility band     |
| 12  | Bollinger Width | (same)                  | Volatility measure  |
| 13  | EMA-9           | period=9                | Short-term trend    |
| 14  | EMA-21          | period=21               | Medium-term trend   |
| 15  | ATR             | period=14               | Volatility          |
| 16  | VWAP            | —                       | Fair value          |
| 17  | OBV             | —                       | Volume trend        |
| 18  | ADX             | period=14               | Trend strength      |
| 19  | Stochastic %K   | k=14, d=3               | Overbought/oversold |
| 20  | Stochastic %D   | (same)                  | Smoothed momentum   |
| 21  | CCI             | period=20               | Mean reversion      |
| 22  | Williams %R     | period=14               | Overbought/oversold |
| 23  | MFI             | period=14               | Volume-weighted RSI |

**Fourier temporal features (5) — captures intraday cyclical patterns:**

| #   | Feature                    | Formula                          | Captures                    |
| --- | -------------------------- | -------------------------------- | --------------------------- |
| 24  | Time-of-day sin (primary)  | `sin(2*pi*t / 390)`              | Open/close volatility cycle |
| 25  | Time-of-day cos (primary)  | `cos(2*pi*t / 390)`              | (same, phase-shifted)       |
| 26  | Time-of-day sin (harmonic) | `sin(4*pi*t / 390)`              | Lunch-dip / mid-day pattern |
| 27  | Time-of-day cos (harmonic) | `cos(4*pi*t / 390)`              | (same, phase-shifted)       |
| 28  | Day-of-week (normalized)   | `day_index / 4.0` (Mon=0, Fri=1) | Weekly seasonality          |

Where `t` = minutes since market open (0–390), `390` = total trading minutes per day.

**Normalization**: Z-score on indicators (fit on train split only). OHLCV replacements and Fourier features are inherently stationary / bounded.
**NaN handling**: Drop warm-up rows where indicators are undefined.

### 1.4 Label Generation (Volatility-Normalized Adaptive Thresholds)

Simple percentile thresholds treat a +0.1% move the same whether the market is calm or turbulent. Volatility-normalization fixes this.

- **Step 1**: Compute 5-minute log returns: `r_t = ln(Close_t / Close_{t-1})`
- **Step 2**: Compute EWMA volatility (RiskMetrics convention):
  `sigma_t^2 = 0.94 * sigma_{t-1}^2 + 0.06 * r_{t-1}^2`
- **Step 3**: Normalize returns: `z_t = r_t / sigma_t`
- **Step 4**: Apply fixed thresholds on normalized returns:
  - `Down (0)`: z_t < -0.5
  - `Neutral (1)`: -0.5 <= z_t <= +0.5
  - `Up (2)`: z_t > +0.5
- Thresholds (±0.5σ) are tunable in config, adjusted for approximate class balance.
- EWMA parameters fitted on training data only, then applied forward (no leakage).

### 1.5 Dataset & Splits

- **Windowing**: Sliding window, sequence length = 60 timesteps (~5 hours of 5m data)
- **Input shape**: `(batch, 60, 28)`
- **Output**: Single class label per window
- **Primary split** — Walk-forward validation (4 folds):
  ```
  Fold 1: Train months 1-7,  Val month 8,   Test month 9
  Fold 2: Train months 2-8,  Val month 9,   Test month 10
  Fold 3: Train months 3-9,  Val month 10,  Test month 11
  Fold 4: Train months 4-10, Val month 11,  Test month 12
  ```
- Report mean ± std of all metrics across folds for statistical rigor.
- **Class weighting**: Inverse-frequency weights (applied in loss function)

---

## 2. Model Architectures

### 2.1 LSTM + Temporal Attention

```
Input (batch, 60, 28)
  → 3-layer Stacked LSTM (hidden=256, dropout=0.3, layer_norm after each layer)
  → Temporal Attention:
      e_t = v^T · tanh(W · h_t + b)         [attention energy per timestep]
      alpha_t = softmax(e_t)                  [attention weights]
      context = Σ(alpha_t · h_t)              [weighted sum over all 60 steps]
  → FC(256 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 3)
```

The attention mechanism learns which timesteps matter most (instead of relying on the last hidden state to carry all information). Attention weights are also interpretable — can be plotted per prediction.

### 2.2 Temporal Fusion Transformer (TFT)

```
Input (batch, 60, 28) + Static covariate (ticker embedding, dim=16)
  → Variable Selection Network (per-feature GLU gating — exposes importance weights)
  → Gated Residual Network blocks (with static covariate context)
  → LSTM Encoder (hidden=128)
  → Multi-Head Attention (4 heads) with gated skip connections
  → FC → 3-class output
```

Key components:

- **VSN**: Produces feature importance scores (Research Q3)
- **Static covariate encoder**: Learned embedding per ticker (30 stocks → 16-dim), conditions GRN blocks on stock identity
- **GLU gating on skip connections**: Faithful to the original Lim et al. (2021) paper

### 2.3 Dilated CNN-LSTM Hybrid

```
Input (batch, 60, 28)  [reshaped to (batch, 28, 60) for Conv1d]
  → Dilated 1D Conv stack (exponentially growing receptive field):
      Conv1d(kernel=3, dilation=1, filters=64) + BatchNorm + ReLU   [RF: 3]
      Conv1d(kernel=3, dilation=2, filters=64) + BatchNorm + ReLU   [RF: 7]
      Conv1d(kernel=3, dilation=4, filters=64) + BatchNorm + ReLU   [RF: 15]
  → Concatenate all conv outputs (192 channels)
  → Permute to (batch, seq, 192)
  → 2-layer Bidirectional LSTM (hidden=128)
  → Last hidden state (256-dim)
  → FC(256 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 3)
```

Dilated convolutions expand the receptive field from 7 → 15 timesteps without adding parameters, capturing longer-range local patterns (TCN/WaveNet principle).

---

## 3. Training Configuration

| Parameter         | Value                                         |
| ----------------- | --------------------------------------------- |
| Loss              | Focal Loss (gamma=2.0, alpha=class_weights)   |
| Label smoothing   | 0.1                                           |
| Optimizer         | AdamW                                         |
| Learning rate     | 1e-3                                          |
| Weight decay      | 1e-5                                          |
| Scheduler         | Linear warmup (10% steps) → CosineAnnealingLR |
| Batch size        | 256                                           |
| Max epochs        | 50                                            |
| Early stopping    | patience=10 on val macro F1                   |
| Gradient clipping | max_norm=1.0                                  |
| Device            | CUDA if available, else CPU                   |

**Focal Loss** (`src/training/losses.py`):

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Down-weights easy examples, focuses training on hard boundary cases (neutral vs up/down). Strictly better than weighted CrossEntropy for ambiguous class boundaries.

**Label smoothing**: Replaces hard [0,0,1] targets with soft [0.033, 0.033, 0.933]. Accounts for the inherent noise at label boundaries — a return at the 34th percentile vs 32nd percentile shouldn't be treated as categorically different.

**LR warmup**: Prevents early gradient explosions, especially important for the TFT's attention layers.

---

## 4. Evaluation & Analysis

### 4.1 Metrics (all 3 models)

- Accuracy, Macro F1-score, per-class precision/recall/F1
- Confusion matrix (3x3)
- Comparison vs naive majority-class baseline
- **McNemar's test** for pairwise statistical significance between models:
  ```
  chi2 = (n01 - n10)^2 / (n01 + n10)
  ```
  Where n01/n10 are discordant prediction counts. p < 0.05 = statistically significant difference.
- Report mean ± std across walk-forward folds

### 4.2 Feature Importance (Research Q3)

- **TFT variable selection**: Extract learned VSN weights from trained model
- **Mutual information**: `MI(X_i; Y) = Σ p(x,y) · log(p(x,y) / (p(x)·p(y)))` via `sklearn.feature_selection.mutual_info_classif` — information-theoretic, non-linear feature ranking
- **LSTM ablation**: Retrain with one feature removed at a time, measure F1 drop
- Compare all three rankings in a single figure

### 4.3 Volatility Regime Analysis (Research Q4)

- Classify trading sessions as high-vol / low-vol using ATR percentiles (top 25% = high-vol)
- Report accuracy and F1 per regime per model
- Compute degradation percentage

### 4.4 Backtesting (Research Q5)

- **Window**: Last 30 trading days of test set
- **Strategy**: Buy on "Up" signal, sell on "Down", hold on "Neutral"
- **Confidence filtering** (MC Dropout): Run 30 forward passes with dropout enabled at inference. Only trade when max softmax probability > 0.6. Compare filtered vs unfiltered performance.
- **Risk metrics**:
  - **Sharpe ratio**: `(R - R_f) / sigma` (annualized)
  - **Sortino ratio**: `(R - R_f) / sigma_downside` (only penalizes downside volatility)
  - **Calmar ratio**: `Annualized return / Max drawdown`
  - **Max drawdown** + max drawdown duration
- **Benchmark**: Buy-and-hold on same period
- **Output**: Equity curve plot, metrics table

---

## 5. Experiment Tracking

- **MLflow** for logging:
  - Hyperparameters
  - Train/val loss and F1 per epoch
  - Final test metrics per walk-forward fold
  - Model artifacts (checkpoints)
- One MLflow experiment per model architecture

---

## 6. Notebooks

| Notebook                    | Purpose                                                                                                    |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `01_data_exploration.ipynb` | EDA, feature distributions, correlation heatmap, class balance, MI feature ranking                         |
| `02_training.ipynb`         | Train all 3 models (walk-forward), learning curves, MLflow comparison                                      |
| `03_evaluation.ipynb`       | Model comparison, McNemar's test, confusion matrices, feature importance, volatility analysis, backtesting |

---

## 7. Next.js Demo App

`frontend/src/app/page.tsx` — interactive dashboard for presenting results:

- **Stock selector**: Pick any of the 30 tickers
- **Model selector**: Choose LSTM / TFT / CNN-LSTM
- **Live prediction view**: Recent price chart with predicted direction overlay
- **Model comparison**: Side-by-side accuracy/F1 table and bar charts
- **Feature importance**: Interactive bar chart from TFT variable selection + MI
- **Backtesting results**: Equity curve, Sharpe/Sortino/Calmar stats
- **Confusion matrices**: Per-model heatmaps
- **Confidence analysis**: Prediction confidence distribution (MC Dropout)

---

## 8. Implementation Order

| Step | What                                                          | Depends On |
| ---- | ------------------------------------------------------------- | ---------- |
| 1    | Scaffolding (dirs, requirements, .gitignore, ruff)            | —          |
| 2    | Data download + caching (yfinance)                            | Step 1     |
| 3    | Feature engineering (stationary OHLCV + indicators + Fourier) | Step 2     |
| 4    | Label generation (volatility-normalized)                      | Step 3     |
| 5    | PyTorch Dataset + walk-forward DataLoaders                    | Step 4     |
| 6    | LSTM + temporal attention model                               | Step 5     |
| 7    | Training pipeline + focal loss + metrics + MLflow             | Step 6     |
| 8    | Train & validate LSTM (end-to-end test)                       | Step 7     |
| 9    | CNN-LSTM (dilated) model + train                              | Step 7     |
| 10   | TFT model + train                                             | Step 7     |
| 11   | Feature importance (MI + VSN + ablation)                      | Step 10    |
| 12   | Volatility analysis                                           | Steps 8-10 |
| 13   | Backtesting (with MC Dropout confidence filtering)            | Steps 8-10 |
| 14   | Notebooks (EDA, training, evaluation)                         | Steps 1-13 |
| 15   | Next.js demo app                                              | Steps 1-13 |

---

## 9. Dependencies

```
# Core
torch>=2.11
numpy>=2.4
pandas>=3.0

# Data
yfinance>=1.2
pandas-ta>=0.3.14b

# ML / Evaluation
scikit-learn>=1.8

# Visualization
matplotlib>=3.10
seaborn>=0.13
plotly>=6.7

# Experiment Tracking
mlflow>=3.11

# Config / Utilities
pyyaml>=6.0
tqdm>=4.67

# Linting
ruff>=0.15
```

---

## 10. Ruff Configuration (in `pyproject.toml`)

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["src"]
```

Rules enabled:

- **E/W**: pycodestyle errors/warnings
- **F**: pyflakes (unused imports, undefined names)
- **I**: isort (import ordering)
- **N**: pep8-naming
- **UP**: pyupgrade (modern Python syntax)
- **B**: bugbear (common pitfalls)
- **SIM**: simplify (code simplification suggestions)

---

## 11. Advanced Techniques Summary

| Technique                     | Where                   | Mathematical Concept                                 |
| ----------------------------- | ----------------------- | ---------------------------------------------------- |
| Log-return OHLCV              | `features.py`           | Stationarity via log-differencing                    |
| Fourier time features         | `features.py`           | Sinusoidal encoding of cyclical patterns             |
| EWMA volatility normalization | `labels.py`             | RiskMetrics exponential weighting (lambda=0.94)      |
| Focal loss                    | `losses.py`             | Down-weighting easy examples: `(1-p_t)^gamma`        |
| Label smoothing               | `trainer.py`            | Soft targets for label noise at boundaries           |
| Temporal attention (LSTM)     | `lstm.py`               | Bahdanau additive attention over hidden states       |
| Dilated convolutions          | `cnn_lstm.py`           | Exponential receptive field growth (WaveNet/TCN)     |
| Variable Selection Network    | `tft.py`                | Learned GLU-gated per-feature importance             |
| Static covariate embedding    | `tft.py`                | Ticker identity conditioning via learned embeddings  |
| MC Dropout uncertainty        | `backtest.py`           | Bayesian approximation via dropout at inference      |
| Mutual information            | `feature_importance.py` | `MI(X;Y) = Σ p(x,y) log(p(x,y)/(p(x)p(y)))`          |
| McNemar's test                | `metrics.py`            | Chi-squared test on discordant predictions           |
| Walk-forward validation       | `dataset.py`            | Rolling temporal CV with expanding/sliding window    |
| Sortino / Calmar ratios       | `backtest.py`           | Downside-risk-adjusted and drawdown-adjusted returns |

---

## Decisions Made

- [x] **Tickers**: 30 stocks, 6 per sector (unchanged)
- [x] **Labels**: Volatility-normalized adaptive thresholds (EWMA + fixed σ cutoffs)
- [x] **Sequence length**: 60 timesteps
- [x] **Features**: 28 total (5 stationary OHLCV + 18 indicators + 5 Fourier/temporal)
- [x] **Demo**: Next.js app
- [x] **Data source**: single configured provider in config
- [x] **Linting**: Ruff with pyproject.toml config
- [x] **Loss**: Focal loss + label smoothing
- [x] **Validation**: Walk-forward (4 folds) + McNemar's significance test
- [x] **Backtesting**: Sharpe + Sortino + Calmar + MC Dropout confidence filtering
