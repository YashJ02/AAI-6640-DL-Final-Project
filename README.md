# AAI 6640 Final Project: Intraday Direction Intelligence Platform

Production-style deep learning system for intraday 3-class direction prediction (`down`, `neutral`, `up`) on a 30-stock S&P 500 universe, with a full Python training pipeline and a Next.js analytics dashboard.

## Table of Contents

1. [Project Overview](#project-overview)
2. [What This Repository Implements](#what-this-repository-implements)
3. [Architecture Summary](#architecture-summary)
4. [Repository Structure](#repository-structure)
5. [Environment Setup](#environment-setup)
6. [Quick Start Commands](#quick-start-commands)
7. [Data Pipeline Details](#data-pipeline-details)
8. [Feature Engineering Details](#feature-engineering-details)
9. [Labeling Strategy](#labeling-strategy)
10. [Dataset and Split Strategy](#dataset-and-split-strategy)
11. [Model Architectures](#model-architectures)
12. [Training Strategy](#training-strategy)
13. [Evaluation and Analysis](#evaluation-and-analysis)
14. [Artifact Contract](#artifact-contract)
15. [Dashboard (Frontend)](#dashboard-frontend)
16. [Config Profiles](#config-profiles)
17. [Reproducibility and Quality Controls](#reproducibility-and-quality-controls)
18. [Current Artifact Snapshot In This Repo](#current-artifact-snapshot-in-this-repo)
19. [Current Model Metrics Snapshot](#current-model-metrics-snapshot)
20. [Developer Commands](#developer-commands)
21. [Known Limitations](#known-limitations)
22. [Contributors](#contributors)
23. [Course Context](#course-context)

## Project Overview

This project compares multiple deep learning architectures for short-horizon intraday direction prediction under realistic market-data constraints.

Primary goals:

- Build a complete data-to-model-to-dashboard workflow (not just a notebook experiment).
- Enforce leakage-resistant preprocessing and fold-aware training behavior.
- Compare model families using both classification and trading-oriented metrics.
- Persist standardized artifacts for reproducibility and frontend visualization.

Core stack:

- Python 3.11+
- PyTorch 2.x
- yfinance + pandas-ta
- scikit-learn / scipy metrics and statistical testing
- MLflow tracking
- Next.js 16 + React 19 + TypeScript dashboard

## What This Repository Implements

- End-to-end intraday data pipeline:
  - Download and cache OHLCV bars from yfinance.
  - Clean bars with session, validity, and anomaly checks.
  - Engineer stationary, technical, and temporal features.
  - Build volatility-normalized labels.
- Time-aware dataset building:
  - Sliding sequence windows.
  - Session-based rolling folds, time-aware k-fold, or month-based splits.
  - Fold-scoped indicator normalization and non-finite value guards.
- Model training and comparison:
  - LSTM + temporal attention.
  - Temporal Fusion Transformer (TFT-style).
  - Dilated CNN + BiLSTM hybrid.
  - Optional weighted soft-voting ensemble if multiple models are trained.
- Evaluation and analytics:
  - Accuracy, macro-F1, confusion matrices, classification report.
  - Majority-class baseline comparison.
  - Pairwise McNemar significance testing.
  - Optional feature importance, volatility regime analysis, and backtesting pipelines.
- Dashboard delivery:
  - Next.js API reads artifacts from parent `artifacts/` directory.
  - Zod schema validation for robust typed payloads.

## Architecture Summary

High-level execution flow:

1. `src/main.py` parses CLI args and loads YAML config.
2. `src/pipeline.py` orchestrates download -> cleaning -> features -> labels.
3. `src/data/dataset.py` builds fold-specific datasets/loaders with leakage controls.
4. `src/training/trainer.py` trains each model across folds with early stopping and checkpointing.
5. `src/pipeline.py` writes standardized artifacts (`results_summary.json`, predictions, checkpoints, data-quality files, KPI report).
6. Frontend API (`frontend/src/app/api/dashboard/route.ts`) reads artifacts and serves dashboard payload.

## Repository Structure

```text
AAI-6640-DL-Final-Project/
|- config/                     # Experiment profile YAMLs
|- artifacts/                  # Generated outputs (checkpoints, predictions, summaries)
|- frontend/                   # Next.js dashboard
|- scripts/                    # Utility scripts (e.g., presentation generation)
|- src/
|  |- data/                    # download, cleaning, features, labels, dataset
|  |- models/                  # lstm, tft, cnn_lstm model definitions
|  |- training/                # trainer, losses, metrics
|  |- evaluation/              # backtest, feature importance, regime analysis
|  |- utils/                   # config and reproducibility helpers
|  |- pipeline.py              # end-to-end Python orchestration
|  |- main.py                  # CLI entrypoint
|- pyproject.toml              # Metadata + Ruff config
|- requirements.txt            # Python dependencies
|- PROJECT_PLAN.md             # Project planning and methodology notes
|- README.md
```

Notes:

- `data/raw/` is created at runtime (cache directory), not committed with market files.
- `mlruns/` is generated when MLflow logging is active.

## Environment Setup

### 1) Python backend

Recommended Python version: `>=3.11`.

```bash
pip install -r requirements.txt
```

If you use a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Frontend dashboard

This repository includes `frontend/bun.lock`, so Bun is the primary package manager.

```bash
cd frontend
bun install
```

You can also use npm if needed:

```bash
cd frontend
npm install
```

## Quick Start Commands

Run from repository root.

### Data preparation only

```bash
python -m src.main --mode data
```

Runs download + cleaning + feature engineering + labeling and writes data-quality artifacts.

### Full training workflow

```bash
python -m src.main --mode full
```

### Train mode (same training pipeline path as `full`)

```bash
python -m src.main --mode train
```

### Train only selected models

```bash
python -m src.main --mode train --models lstm tft
```

### Use a different config profile

```bash
python -m src.main --config config/kfold_cv.yaml --mode full
```

### Force fresh download (ignore cache)

```bash
python -m src.main --mode full --force-refresh
```

### Start dashboard

```bash
cd frontend
bun run dev
```

Open `http://localhost:3000`.

## Data Pipeline Details

### Universe

Default equity universe is 30 S&P 500 tickers grouped by sector in config:

- technology
- financials
- healthcare
- energy
- consumer_discretionary

Related market/context symbols (default):

- `SPY`, `QQQ`, `^VIX`, `IWM`, `DIA`, `TLT`, `GLD`, `DX-Y.NYB`

### Download and cache behavior (`src/data/download.py`)

- Uses yfinance intraday bars with project interval notation (`1Min`, `5Min`, `15Min`).
- Provider-aware lookback caps:
  - `1Min` capped to 7 days.
  - `5Min`/`15Min` capped to 60 days.
- Cache key pattern:
  - Main ticker: `{ticker}_{interval}_{history_days}d.parquet`
  - Related symbol: `related_{symbol}_{interval}_{history_days}d.parquet`
- Retries downloads up to 3 times.
- Related symbol failures are skipped with warning (pipeline continues).

### Cleaning rules (`src/data/cleaning.py`)

For each ticker:

1. Drop duplicate timestamps.
2. Keep only regular session bars in `[start, end)` (default 09:30 to 16:00 NY).
3. Remove invalid OHLCV rows (non-positive prices, invalid bar geometry, optional zero-volume drop).
4. Remove extreme anomalies:
   - absolute log return above configured threshold
   - intrabar range above configured threshold
5. Drop low-coverage sessions below `min_session_coverage`.

Detailed per-ticker cleaning stats are written to CSV artifacts.

## Feature Engineering Details

Base feature set has 28 columns from three blocks.

### 1) Stationary OHLCV transforms (5)

- `log_return`
- `hl_range`
- `oc_body`
- `upper_shadow`
- `volume_log_change`

### 2) Technical indicators (18)

- RSI
- MACD line/signal/hist
- Bollinger upper/lower/width
- EMA 9/21
- ATR
- VWAP
- OBV
- ADX
- Stochastic K/D
- CCI
- Williams %R
- MFI

### 3) Temporal/Fourier features (5)

- Primary intraday sin/cos cycle
- Harmonic intraday sin/cos cycle
- Normalized day-of-week

### Related context features

When `data.use_related_features=true`, related symbols add:

- per-symbol return feature (`*_ret1`)
- per-symbol volume-change feature (`*_volchg1`)
- optional realized volatility feature (`*_rv12`) if enabled

Total feature count is dynamic:

- `28 + related columns successfully available`

### Non-finite handling

- Feature builders sanitize `+/-inf` to `NaN` after log/division ops.
- Before sequence construction, fold frames drop non-finite rows across active feature columns and label.

## Labeling Strategy

Label generation is implemented in `src/data/labels.py`.

Given close price `C_t` and horizon `h` (default one bar):

$$
r_t = \log\left(\frac{C_{t+h}}{C_t}\right)
$$

Session boundary protection:

- Future returns that cross to a different trading day are masked out.

EWMA variance recursion:

$$
\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1-\lambda)r_{t-1}^2
$$

Normalized return:

$$
z_t = \frac{r_t}{\sigma_t}
$$

Class mapping:

- `down` (0): `z_t < threshold_down`
- `neutral` (1): `threshold_down <= z_t <= threshold_up`
- `up` (2): `z_t > threshold_up`

Optional fold-adaptive thresholding (`src/data/dataset.py`):

- Train split quantiles can override fixed thresholds for each fold.
- Falls back to configured fixed thresholds if quantile thresholds are invalid.

## Dataset and Split Strategy

### Sequence dataset

- Sequence length: configurable (default 60 timesteps).
- Each sample is `(X_seq, y_end, ticker_id)`.
- Windows never cross ticker boundaries.

### Split modes (`dataset.split_mode`)

1. `sessions` (default): rolling session-based train/val/test windows.
2. `kfold`: session-level k-fold with optional time-aware train-val pool.
3. `month`: explicit month-index folds from config.

### Leakage controls in folds

- Indicator normalization stats are fit on train split only and applied to val/test.
- Fold-adaptive labels derive thresholds from train split only.
- Non-finite rows are removed before DataLoader creation.

### Class balancing

- Optional weighted train sampler (`dataset.weighted_sampler`).
- Class weights computed from train labels and passed to focal loss (unless disabled).

## Model Architectures

Implemented in `src/models/`.

### 1) LSTM + Temporal Attention (`lstm.py`)

- Stacked LSTM encoder.
- LayerNorm on sequence outputs.
- Additive temporal attention for context vector.
- MLP classifier head.

### 2) Temporal Fusion Transformer style model (`tft.py`)

- Variable selection gating per timestep.
- Ticker embedding as static covariate.
- Gated Residual Network blocks.
- LSTM encoder + multi-head self-attention + gated skip.
- Classifier head from final fused timestep.

### 3) Dilated CNN + BiLSTM hybrid (`cnn_lstm.py`)

- Parallel dilated Conv1d branches.
- Concatenated conv features.
- Bidirectional LSTM sequence modeling.
- MLP classifier head.

### Optional ensemble

If at least two model families are trained, pipeline can build weighted soft-voting ensemble from fold probabilities.

## Training Strategy

Training engine lives in `src/training/trainer.py`.

Key components:

- Loss: multi-class focal loss with optional class weighting and label smoothing.
- Optimizer: AdamW.
- LR schedule: linear warmup + cosine decay (`LambdaLR`).
- Early stopping on validation macro-F1.
- Optional AMP on CUDA.
- Optional `torch.compile`.
- Gradient clipping.
- Fold-level checkpointing to `artifacts/checkpoints/`.
- Optional decision bias tuning on validation probabilities (`down_bias`, `up_bias`) optimizing macro-F1 or accuracy.
- Optional MLflow logging per fold.

## Evaluation and Analysis

### Always produced by training pipeline

- Test loss, accuracy, macro-F1.
- Confusion matrix and classification report.
- Majority-class baseline metrics.
- Pairwise McNemar test per fold.
- Fold summaries (mean/std) in `results_summary.json`.

### Additional analysis utilities (`src/evaluation/`)

- `feature_importance.py`:
  - mutual information ranking
  - TFT variable selection weights
  - optional LSTM ablation ranking
- `volatility_analysis.py`:
  - low-vol vs high-vol regime metrics using ATR percentile split
  - degradation percentages
- `backtest.py`:
  - class-to-position mapping
  - confidence-threshold gating
  - portfolio return aggregation for multi-ticker frames
  - Sharpe/Sortino/Calmar/max-drawdown metrics

The orchestration helpers are in `src/evaluation/pipeline.py` and can be invoked from custom scripts/notebooks.

## Artifact Contract

### Core artifacts from `python -m src.main --mode train/full`

- `artifacts/results_summary.json`
- `artifacts/kpi_accuracy_report.json`
- `artifacts/data_quality/modeling_summary.json`
- `artifacts/data_quality/ticker_cleaning_report.csv`
- `artifacts/data_quality/ticker_modeling_report.csv`
- `artifacts/checkpoints/<model>_fold_<id>.pt`
- `artifacts/predictions/<model>_fold_<id>.npz`
- `artifacts/training_logs/<model>_fold_<id>_history.csv` (if enabled)

### Optional dashboard-enrichment artifacts

If additional evaluation/reporting scripts are run, dashboard also consumes:

- `artifacts/overfit_health_report.json`
- `artifacts/evaluation_accuracy90/<model>/evaluation_summary.json`
- `artifacts/evaluation_accuracy90/<model>/volatility_regime_metrics.csv`
- `artifacts/evaluation_accuracy90/<model>/backtest_unfiltered_curve.csv`
- `artifacts/evaluation_accuracy90/<model>/backtest_filtered_curve.csv`
- `artifacts/evaluation_accuracy90/<model>_threshold_grid.csv`
- `artifacts/evaluation_accuracy90/sharpe_optimization_summary.json`

If these optional files are missing, frontend falls back to empty/null sections for those views.

## Dashboard (Frontend)

### How it loads data

- API endpoint: `GET /api/dashboard`
- Route handler: `frontend/src/app/api/dashboard/route.ts`
- Server artifact loader: `frontend/src/lib/server/artifacts.ts`
- Artifact root resolution: frontend resolves parent project root and reads `../artifacts`

### Dashboard coverage

- Model comparison and key metrics
- Confusion matrix and decision bias summary
- KPI gate summary
- Data quality cards and ticker reports
- Training history charts
- McNemar matrices
- Optional threshold sweep, volatility regime, and backtest views

### Run commands

```bash
cd frontend
bun run dev
```

For production build:

```bash
cd frontend
bun run build
bun run start
```

## Config Profiles

Profiles are in `config/`.

| Profile | Main Purpose | Key Differences |
| --- | --- | --- |
| `default.yaml` | Balanced baseline | Session split, adaptive thresholds on, weighted sampler on, up to 2 folds, 45 epochs |
| `high_accuracy.yaml` | Faster focused run | 1 session fold, 20 epochs, adaptive thresholds on |
| `kfold_cv.yaml` | Time-aware cross-validation | `split_mode: kfold`, `n_splits: 5`, `time_aware: true`, `max_folds: 3` |
| `accuracy_90.yaml` | Accuracy-targeted profile | Wide label thresholds (`-2/+2`), adaptive thresholds off, class weights off, decision tuning metric set to accuracy |
| `kpi_dual.yaml` | Enforced KPI gates | KPI enforcement with minimum accuracy and delta-vs-baseline thresholds |
| `benchmark_lstm_v2.yaml` | Benchmark profile scaffold | Similar baseline settings; intended for benchmark-specific iterations |

Use any profile via:

```bash
python -m src.main --config config/<profile>.yaml --mode full
```

## Reproducibility and Quality Controls

- Seed management (`numpy`, `torch`, optional deterministic CuDNN).
- Cache-first data access with explicit force-refresh option.
- Session-safe label horizon handling.
- Train-only normalization and optional train-only adaptive thresholds.
- Non-finite feature row filtering before sequence generation.
- Artifact-first outputs for auditability.

## Current Artifact Snapshot In This Repo

At the time of this README rewrite, checked-in artifacts include:

- `artifacts/results_summary.json`
- `artifacts/checkpoints/` for all three model families and two folds
- `artifacts/predictions/` for all three model families and two folds
- `artifacts/data_quality/modeling_summary.json` and ticker-level CSV reports

Current data-quality summary file reports:

- rows: `133660`
- tickers: `30`
- feature count: `34` (28 base + 6 related features in this run)

Exact values will change after each new run.

## Current Model Metrics Snapshot

Source: `artifacts/results_summary.json` currently checked into this repository.

| Model | Test Loss (mean ± std) | Test Accuracy (mean ± std) | Test Macro-F1 (mean ± std) | Baseline Accuracy (mean) | Delta Accuracy vs Baseline | Delta Macro-F1 vs Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| `lstm` | 0.4306 ± 0.0102 | 0.4577 ± 0.0190 | 0.3678 ± 0.0019 | 0.5189 | -0.0612 | +0.1402 |
| `cnn_lstm` | 0.4299 ± 0.0092 | 0.4445 ± 0.0146 | 0.3775 ± 0.0005 | 0.5189 | -0.0743 | +0.1499 |
| `tft` | 0.4293 ± 0.0102 | 0.4521 ± 0.0236 | 0.3765 ± 0.0102 | 0.5189 | -0.0668 | +0.1489 |

Interpretation notes:

- Accuracy is below the majority-class baseline in this snapshot, so raw directional hit-rate is not yet outperforming the naive classifier.
- Macro-F1 is substantially above baseline for all three models, indicating better class balance behavior than majority-only predictions.
- Best macro-F1 in this run is `cnn_lstm` (0.3775 mean), while best accuracy is `lstm` (0.4577 mean).

## Developer Commands

Backend linting:

```bash
ruff check .
```

Frontend linting:

```bash
cd frontend
bun run lint
```

CLI help:

```bash
python -m src.main --help
```

## Known Limitations

- yfinance intraday history windows are provider-limited.
- Related symbols may intermittently fail download and be skipped.
- Advanced evaluation artifacts (threshold sweeps/backtest summary files) are not produced by `src.main` directly and require additional evaluation pipeline execution.
- Performance is sensitive to label-threshold choices and class balance in intraday regimes.

## Contributors

- Ruthvik Bandari
- Om Patel
- Yash Jain

## Course Context

- Course: AAI 6640 - Applied Deep Learning
- Framework: PyTorch 2.x
- Tracking: MLflow
- Linting: Ruff
