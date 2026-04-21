# AAI 6640 Final Project: Intraday Direction Prediction

This repository implements a full end-to-end comparative study for intraday stock direction prediction using three deep learning architectures:

- LSTM + Temporal Attention
- Temporal Fusion Transformer (TFT)
- Dilated CNN-LSTM Hybrid

The implementation follows the complete plan in `PROJECT_PLAN.md`, including:

- Data download and caching (single selected provider via config)
- Related market-context ingestion (SPY/QQQ/VIX by default)
- Data quality cleaning and per-ticker audit reports
- 28-feature engineering pipeline
- Volatility-normalized 3-class label generation
- Walk-forward validation
- Focal loss training with label smoothing and warmup+cosine scheduling
- Significance testing (McNemar)
- Feature importance, volatility regime analysis, and backtesting
- Next.js dashboard for presentation

## Project Structure

```text
AAI-6640-DL-Final-Project/
|-- config/
|   |-- default.yaml
|-- data/
|   |-- raw/
|-- src/
|   |-- data/
|   |   |-- download.py
|   |   |-- features.py
|   |   |-- labels.py
|   |   |-- dataset.py
|   |-- models/
|   |   |-- lstm.py
|   |   |-- tft.py
|   |   |-- cnn_lstm.py
|   |-- training/
|   |   |-- losses.py
|   |   |-- metrics.py
|   |   |-- trainer.py
|   |-- evaluation/
|   |   |-- feature_importance.py
|   |   |-- volatility_analysis.py
|   |   |-- backtest.py
|   |   |-- pipeline.py
|   |-- utils/
|   |   |-- config.py
|   |-- pipeline.py
|   |-- main.py
|-- frontend/
|   |-- src/
|   |-- public/
|   |-- package.json
|   |-- bun.lock
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_training.ipynb
|   |-- 03_evaluation.ipynb
|-- artifacts/
|   |-- checkpoints/
|-- pyproject.toml
|-- requirements.txt
|-- PROJECT_PLAN.md
```

## Setup

1. Create a Python 3.11+ environment.
1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. No API credentials are required for the default open-data setup (`yfinance`).

Frontend setup (Bun runtime):

```bash
cd frontend
bun install
```

### GPU/Performance Defaults

The default config is tuned for consumer NVIDIA GPUs (including RTX 3050 laptops):

- mixed precision (`training.use_amp: true`)
- multi-worker data loading (`dataset.num_workers: 2`)
- pinned-memory transfer (`dataset.pin_memory: true`)
- optional deterministic toggle (`experiment.deterministic`)
- validation-based class decision tuning (`training.decision_tuning`)

### Data Source Notes

- `yfinance` intraday intervals have provider lookback limits (notably around 60 days for 5-minute bars).
- The default `history_days` is set to stay within practical provider limits while maximizing available data.
- Public related-market context symbols are enabled by default (broad indices, sector ETFs, credit/commodities, and optional crypto proxy).
- Related-symbol ingestion is fault-tolerant: symbols with transient download failures are skipped without stopping the full pipeline.

## Execution Guide (Plan-Aligned)

### Step 1: Scaffolding and Config

- Complete in repository via:
  - `config/default.yaml`
  - `requirements.txt`
  - `pyproject.toml`
  - `.gitignore`

### Step 2-5: Data Pipeline

Run data-only pipeline:

```bash
python -m src.main --mode data
```

What this executes:

1. Download/cache OHLCV bars per ticker.
2. Download related-context symbols defined in config.
3. Clean/validate bars and write audit artifacts to `artifacts/data_quality/`.
4. Engineer model features.
5. Build EWMA volatility-normalized labels (session-aware; no overnight label leakage).
6. Construct walk-forward-ready supervised table.

### Step 6-10: Train All Architectures

Run full training pipeline:

```bash
python -m src.main --mode full
```

Optional: train subset of models:

```bash
python -m src.main --mode train --models lstm tft
```

Outputs:

- Checkpoints in `artifacts/checkpoints/`
- Fold predictions in `artifacts/predictions/`
- Epoch/fold training logs in `artifacts/training_logs/`
- Aggregated summary in `artifacts/results_summary.json`

### Step 11-13: Evaluation

Use module utilities in `src/evaluation/pipeline.py` to run:

1. Feature importance suite (MI, optional TFT VSN, optional LSTM ablation)
2. Volatility regime analysis (high-vol vs low-vol)
3. Backtesting (unfiltered and confidence-filtered)

Notebook-driven execution:

- `notebooks/03_evaluation.ipynb`

### Step 14: Notebooks

- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_training.ipynb`
- `notebooks/03_evaluation.ipynb`

Each notebook is organized with explicit step comments for reproducibility.

### Step 15: Next.js Demo

Launch dashboard:

```bash
cd frontend
bun run dev
```

The app reads artifacts from `artifacts/` and shows:

- Model comparison (accuracy/F1)
- Confusion matrices
- Feature importance
- Price + prediction overlays (if prediction timeline exists)
- Confidence distribution
- Backtesting summaries

## Ruff Linting

Lint codebase:

```bash
ruff check .
```

Config is defined in `pyproject.toml`.

## Notes on Comments and Readability

All implementation modules include concise inline comments that explain each major step in the pipeline and model flow.

## Course Context

- Course: AAI 6640 - Applied Deep Learning
- Framework: PyTorch 2.x
- Tracking: MLflow
- Linting: Ruff
