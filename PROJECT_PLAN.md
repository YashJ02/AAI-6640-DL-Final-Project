# Intraday Stock Price Prediction: Multi-Architecture Comparative Study

## Overview

Compare three deep learning architectures (LSTM, TFT, CNN-LSTM) on predicting intraday stock price direction (up/down/neutral) using OHLCV data + technical indicators for 30 S&P 500 equities.

**Course**: AAI 6640 — Applied Deep Learning  
**Framework**: PyTorch 2.x  
**Tracking**: MLflow

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
│   │   ├── features.py             # 12 technical indicators (pandas-ta)
│   │   ├── labels.py               # 3-class label generation
│   │   └── dataset.py              # PyTorch Dataset, windowing, splits
│   ├── models/
│   │   ├── lstm.py                 # Stacked LSTM (baseline)
│   │   ├── tft.py                  # Temporal Fusion Transformer
│   │   └── cnn_lstm.py             # CNN-LSTM Hybrid
│   ├── training/
│   │   ├── trainer.py              # Training loop, early stopping, checkpoints
│   │   └── metrics.py              # Accuracy, F1, confusion matrix
│   ├── evaluation/
│   │   ├── backtest.py             # Portfolio sim, Sharpe ratio
│   │   ├── feature_importance.py   # TFT variable selection + LSTM ablation
│   │   └── volatility_analysis.py  # High-vol vs low-vol breakdown
│   └── utils/
│       └── config.py               # YAML config loader
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 1. Data Pipeline

### 1.1 Tickers (30 stocks, 6 per sector)

| Sector                 | Tickers                          |
|------------------------|----------------------------------|
| Technology             | AAPL, MSFT, NVDA, GOOGL, META, AMZN |
| Financials             | JPM, BAC, GS, MS, WFC, C        |
| Healthcare             | JNJ, UNH, PFE, ABT, MRK, TMO    |
| Energy                 | XOM, CVX, COP, SLB, EOG, MPC    |
| Consumer Discretionary | TSLA, HD, NKE, MCD, SBUX, TJX   |

### 1.2 Data Source

- **Library**: `yfinance`
- **Intervals**: 1-minute, 5-minute, 15-minute
- **Limitation**: yfinance caps 5m data at ~60 trading days, 1m at ~7 days. We will download the maximum available window and document this as a scope constraint. Primary analysis uses **5-minute** interval.
- **Storage**: Cache as parquet files in `data/raw/`

### 1.3 Features (5 OHLCV + 12 Technical Indicators = 17 total)

| #  | Feature             | Source     | Parameters       |
|----|---------------------|------------|------------------|
| 1  | Open                | OHLCV      | —                |
| 2  | High                | OHLCV      | —                |
| 3  | Low                 | OHLCV      | —                |
| 4  | Close               | OHLCV      | —                |
| 5  | Volume              | OHLCV      | —                |
| 6  | RSI                 | pandas-ta  | period=14        |
| 7  | MACD Line           | pandas-ta  | fast=12, slow=26, signal=9 |
| 8  | MACD Signal         | pandas-ta  | (same)           |
| 9  | MACD Histogram      | pandas-ta  | (same)           |
| 10 | Bollinger Upper     | pandas-ta  | period=20, std=2 |
| 11 | Bollinger Lower     | pandas-ta  | (same)           |
| 12 | Bollinger Width     | pandas-ta  | (same)           |
| 13 | EMA-9               | pandas-ta  | period=9         |
| 14 | EMA-21              | pandas-ta  | period=21        |
| 15 | ATR                 | pandas-ta  | period=14        |
| 16 | VWAP                | pandas-ta  | —                |
| 17 | OBV                 | pandas-ta  | —                |

**Normalization**: Per-stock z-score (fit on train split only, applied to val/test).  
**NaN handling**: Drop warm-up rows where indicators are undefined.

### 1.4 Label Generation

- **Target**: Direction of next 5-minute close price change
- **Classes**:
  - `Up (2)`: return > +0.05%
  - `Neutral (1)`: return between -0.05% and +0.05%
  - `Down (0)`: return < -0.05%
- Threshold is tunable in config — will adjust based on actual return distribution to achieve reasonable class balance.

### 1.5 Dataset & Splits

- **Windowing**: Sliding window, sequence length = 60 timesteps (~5 hours of 5m data)
- **Input shape**: `(batch, 60, 17)`
- **Output**: Single class label per window
- **Train/Val/Test split** (temporal, no shuffling):
  - Train: first 70% of trading days
  - Validation: next 15%
  - Test: final 15%
- **Class weighting**: Inverse-frequency weights for CrossEntropyLoss

---

## 2. Model Architectures

### 2.1 LSTM (Baseline)

```
Input (batch, 60, 17)
  → 3-layer Stacked LSTM (hidden=256, dropout=0.3)
  → Last hidden state
  → FC(256 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 3)
  → Softmax
```

### 2.2 Temporal Fusion Transformer (TFT)

```
Input (batch, 60, 17)
  → Variable Selection Network (per-feature gating — exposes importance weights)
  → Gated Residual Network blocks
  → LSTM Encoder (hidden=128)
  → Multi-Head Attention (4 heads)
  → FC → 3-class output
```

Key: The VSN produces feature importance scores used to answer Research Question 3.

### 2.3 CNN-LSTM Hybrid

```
Input (batch, 60, 17)  [reshaped to (batch, 17, 60) for Conv1d]
  → Parallel 1D Conv branches:
      Conv1d(kernel=3, filters=64) + BatchNorm + ReLU
      Conv1d(kernel=5, filters=64) + BatchNorm + ReLU
      Conv1d(kernel=7, filters=64) + BatchNorm + ReLU
  → Concatenate along channel dim (192 channels)
  → Permute back to (batch, seq, 192)
  → 2-layer Bidirectional LSTM (hidden=128)
  → Last hidden state (256-dim)
  → FC(256 → 128) → ReLU → Dropout(0.3)
  → FC(128 → 3)
```

---

## 3. Training Configuration

| Parameter          | Value                    |
|--------------------|--------------------------|
| Loss               | CrossEntropyLoss (class-weighted) |
| Optimizer          | AdamW                    |
| Learning rate      | 1e-3                     |
| Weight decay       | 1e-5                     |
| Scheduler          | CosineAnnealingLR        |
| Batch size         | 256                      |
| Max epochs         | 50                       |
| Early stopping     | patience=10 on val macro F1 |
| Gradient clipping  | max_norm=1.0             |
| Device             | CUDA if available, else CPU |

---

## 4. Evaluation & Analysis

### 4.1 Metrics (all 3 models)
- Accuracy
- Macro F1-score
- Per-class precision, recall, F1
- Confusion matrix (3x3)
- Comparison vs naive majority-class baseline

### 4.2 Feature Importance (Research Q3)
- **TFT**: Extract learned variable selection weights from VSN
- **LSTM ablation**: Retrain with one feature removed at a time, measure F1 drop
- Output: Ranked bar chart of feature importance

### 4.3 Volatility Regime Analysis (Research Q4)
- Classify sessions as high-vol / low-vol using ATR percentiles (top 25% = high-vol)
- Report accuracy and F1 per regime per model
- Compute degradation percentage

### 4.4 Backtesting (Research Q5)
- **Window**: Last 30 trading days of test set
- **Strategy**: Buy on "Up" signal, sell on "Down", hold on "Neutral"
- **Metrics**: Cumulative return, annualized Sharpe ratio, max drawdown
- **Benchmark**: Buy-and-hold on same period
- **Output**: Equity curve plot

---

## 5. Experiment Tracking

- **MLflow** for logging:
  - Hyperparameters
  - Train/val loss and F1 per epoch
  - Final test metrics
  - Model artifacts (checkpoints)
- One MLflow experiment per model architecture

---

## 6. Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Load data, EDA, feature distributions, correlation heatmap, class balance |
| `02_training.ipynb` | Train all 3 models, learning curves, MLflow comparison |
| `03_evaluation.ipynb` | Model comparison table, confusion matrices, feature importance, volatility analysis, backtesting |

---

## 7. Implementation Order

| Step | What | Depends On |
|------|------|------------|
| 1 | Project scaffolding (dirs, requirements.txt, .gitignore, config) | — |
| 2 | Data download + caching | Step 1 |
| 3 | Feature engineering + label generation | Step 2 |
| 4 | PyTorch Dataset + DataLoaders | Step 3 |
| 5 | LSTM model | Step 4 |
| 6 | Training pipeline + metrics + MLflow | Step 5 |
| 7 | Train & validate LSTM (end-to-end test) | Step 6 |
| 8 | CNN-LSTM model + train | Step 6 |
| 9 | TFT model + train | Step 6 |
| 10 | Feature importance analysis | Step 9 |
| 11 | Volatility analysis | Step 7-9 |
| 12 | Backtesting engine | Step 7-9 |
| 13 | Notebooks (EDA, training, evaluation) | Steps 1-12 |

---

## 8. Dependencies

```
torch>=2.0
yfinance>=0.2.31
pandas-ta>=0.3.14
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
mlflow>=2.9
pyyaml>=6.0
tqdm>=4.65
```

---

## Open Questions / Things to Refine

- [ ] Should we adjust the 30 tickers? Any specific ones you want in/out?
- [ ] Label threshold (0.05%) — want to keep this or adjust?
- [ ] Sequence length (60 timesteps) — change?
- [ ] Any additional technical indicators beyond the 12 listed?
- [ ] Do you want a Streamlit/Gradio demo, or notebooks only?
- [ ] yfinance 60-day limit for 5m data — acceptable, or should we explore alternative data sources?
