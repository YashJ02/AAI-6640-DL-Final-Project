# AAI 6640 Final Project — Intraday Direction Intelligence Platform

Production-style deep learning system for **intraday 3-class direction prediction** (`down`, `neutral`, `up`) on a 30-stock S&P 500 universe.

This repository includes:

- End-to-end data, feature, labeling, training, and evaluation pipeline (PyTorch)
- Three comparative model families:
  - **LSTM + Temporal Attention**
  - **Temporal Fusion Transformer (TFT)**
  - **Dilated CNN-LSTM Hybrid**
- Robust data-quality auditing and fold-aware normalization
- Statistical and trading-oriented evaluation (McNemar, volatility regime analysis, backtesting)
- Next.js analytics dashboard consuming generated artifacts

---

## 1) Scope and Objectives

This project is built as a full-stack research-to-delivery workflow for financial ML under realistic constraints:

- Open-data ingestion via `yfinance`
- Intraday bars (`5Min` default) with provider-aware history limits
- Leakage controls (session-safe label generation, fold-scoped normalization)
- Time-aware split strategies (`sessions`, `kfold`, `month`)
- Risk-adjusted model assessment (Sharpe/Sortino/Calmar + drawdown)

Primary outcome: compare model families on directional classification quality and downstream strategy behavior, while preserving reproducibility and auditability.

---

## 2) System Architecture

### Core pipeline stages

1. **Download & Cache**
   - Universe + related-market symbols cached to parquet in `data/raw`
2. **Cleaning & Session Validation**
   - Non-session bars, malformed OHLCV, outliers, and low-coverage sessions removed
3. **Feature Engineering (28 base features)**
   - 5 stationary OHLCV transforms
   - 18 technical indicators
   - 5 Fourier/temporal cyclic features
   - Optional related-symbol context features
4. **Label Generation**
   - EWMA volatility-normalized future return labels (`down/neutral/up`)
5. **Dataset Construction**
   - Sliding windows (default sequence length = 60)
   - Fold generation using configurable split mode
6. **Training & Selection**
   - Focal loss, warmup+cosine LR, early stopping, AMP
   - Optional class-decision bias tuning on validation fold
7. **Evaluation**
   - Classification metrics, McNemar significance
   - Feature-importance suite
   - Volatility-regime degradation
   - Backtesting (filtered/unfiltered)

---

## 3) Universe, Labels, and Features

### Ticker universe (30)

- Technology: AAPL, MSFT, NVDA, GOOGL, META, AMZN
- Financials: JPM, BAC, GS, MS, WFC, C
- Healthcare: JNJ, UNH, PFE, ABT, MRK, TMO
- Energy: XOM, CVX, COP, SLB, EOG, MPC
- Consumer Discretionary: TSLA, HD, NKE, MCD, SBUX, TJX

### Related-market symbols (default)

`SPY`, `QQQ`, `^VIX`, `IWM`, `DIA`, `TLT`, `GLD`, `DX-Y.NYB`

### Labeling method

Let future return be:

$$
r_t = \log\left(\frac{C_{t+h}}{C_t}\right)
$$

EWMA variance recursion:

$$
\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2
$$

Normalized return:

$$
z_t = \frac{r_t}{\sigma_t}
$$

Class mapping (default):

- `down` (0): $z_t < -0.5$
- `neutral` (1): $-0.5 \le z_t \le 0.5$
- `up` (2): $z_t > 0.5$

Optional fold-adaptive thresholds are supported through config.

### Feature blocks

- **Stationary OHLCV transforms (5)**: log-return, range/body/shadow ratios, volume log-change
- **Technical indicators (18)**: RSI, MACD family, Bollinger family, EMA(9/21), ATR, VWAP, OBV, ADX, Stoch, CCI, Williams %R, MFI
- **Temporal/Fourier (5)**: intraday sin/cos (primary + harmonic), normalized day-of-week
- **Related context (optional)**: per-symbol return and volume-change channels

---

## 4) Repository Layout

```text
AAI-6640-DL-Final-Project/
├── config/                   # Experiment profiles (default, KPI, k-fold, accuracy-focused, etc.)
├── data/raw/                 # Cached market parquet files
├── src/
│   ├── data/                 # download, cleaning, features, labels, dataset
│   ├── models/               # lstm.py, tft.py, cnn_lstm.py
│   ├── training/             # losses, metrics, trainer
│   ├── evaluation/           # feature importance, volatility analysis, backtest
│   ├── pipeline.py           # end-to-end orchestration
│   └── main.py               # CLI entrypoint
├── artifacts/                # outputs: checkpoints, logs, summaries, evaluation assets
├── notebooks/                # EDA, training, evaluation notebooks
├── frontend/                 # Next.js dashboard over artifacts/
├── PROJECT_PLAN.md
├── requirements.txt
└── pyproject.toml
```

---

## 5) Environment Setup

### Python (recommended 3.11+)

```bash
pip install -r requirements.txt
```

No API key is required for the default pipeline.

### Frontend (Bun)

```bash
cd frontend
bun install
```

---

## 6) Runbook (CLI)

### Data-only build

```bash
python -m src.main --mode data
```

Executes download → clean → features → labels and writes data quality outputs.

### Full train-and-compare run

```bash
python -m src.main --mode full
```

### Train selected models only

```bash
python -m src.main --mode train --models lstm tft
```

### Use alternate config profile

```bash
python -m src.main --config config/kfold_cv.yaml --mode full
```

### Force data refresh

```bash
python -m src.main --mode full --force-refresh
```

---

## 7) Config Profiles

Common profiles available in `config/`:

- `default.yaml`: balanced baseline with session splits and adaptive thresholds
- `kfold_cv.yaml`: time-aware K-fold validation mode
- `high_accuracy.yaml`: tuned short-run profile
- `accuracy_90.yaml`: stricter thresholding and accuracy-optimized tuning
- `kpi_dual.yaml`: KPI gates enabled (`target_accuracy_min`, `delta_vs_baseline`)
- `benchmark_lstm_v2.yaml`: benchmark-oriented LSTM experiment profile

---

## 8) Artifact Contract (for backend + dashboard)

The dashboard and analysis stack expect artifacts under `artifacts/`, including:

- `artifacts/results_summary.json`
- `artifacts/kpi_accuracy_report.json` (if KPI enabled)
- `artifacts/overfit_health_report.json`
- `artifacts/data_quality/modeling_summary.json`
- `artifacts/data_quality/ticker_modeling_report.csv`
- `artifacts/data_quality/ticker_cleaning_report.csv`
- `artifacts/training_logs/*_history.csv`
- `artifacts/evaluation_accuracy90/**` (evaluation/backtest payloads)

Checkpoint outputs are written to:

- `artifacts/checkpoints/`

---

## 9) Evaluation Coverage

- **Classification**: Accuracy, Macro-F1, class reports, confusion matrix
- **Statistical significance**: pairwise McNemar tests
- **Feature importance**:
  - Mutual information ranking
  - TFT variable selection weights (when model/loader provided)
  - Optional LSTM ablation
- **Regime analysis**: low-vol vs high-vol degradation (ATR percentile partition)
- **Backtesting**:
  - Position mapping from class outputs
  - Optional confidence gating
  - Risk metrics: annualized return/volatility, Sharpe, Sortino, Calmar, max drawdown

---

## 10) Frontend Dashboard

Run the app:

```bash
cd frontend
bun run dev
```

Open `http://localhost:3000`.

The dashboard loads artifacts through the Next.js API route and validates payloads with Zod.

---

## 11) Reproducibility and Quality Controls

- Seed control + deterministic toggle in config
- Fold-scoped normalization and adaptive thresholding
- Session boundary checks in target generation to avoid overnight leakage
- Cached data for reproducible reruns
- Linting via Ruff:

```bash
ruff check .
```

---

## 12) Current Repository Notes

- Existing checkpoint artifact currently present: `artifacts/checkpoints/tft_fold_1.pt`
- Full result/evaluation JSON/CSV assets are generated after executing training/evaluation runs

---

## 13) Contributions

Team contribution breakdown:

- **Ruthvik Bandari**
   - Project idea, ideation, and project description
   - Final training strategy decisions
   - Model cross-fold validation and final decision support (jointly with Om)

- **Om Patel**
   - Complete frontend implementation
   - Model validation workflow
   - Final training and cross-fold validation decisions (jointly with Ruthvik)

- **Yash Jain**
   - Dataset collection
   - Project skeleton setup

---

## 14) Course Context

- Course: **AAI 6640 — Applied Deep Learning**
- Framework: **PyTorch 2.x**
- Tracking: **MLflow**
- Linting: **Ruff**
