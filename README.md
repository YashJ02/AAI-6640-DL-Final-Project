# AAI 6640 Final Project: Intraday Direction Prediction

This repository implements a full end-to-end comparative study for intraday stock direction prediction using three deep learning architectures:

- LSTM + Temporal Attention
- Temporal Fusion Transformer (TFT)
- Dilated CNN-LSTM Hybrid

The implementation follows the complete plan in `PROJECT_PLAN.md`, including:

- Data download and caching (single selected provider via config)
- 28-feature engineering pipeline
- Volatility-normalized 3-class label generation
- Walk-forward validation
- Focal loss training with label smoothing and warmup+cosine scheduling
- Significance testing (McNemar)
- Feature importance, volatility regime analysis, and backtesting
- Streamlit dashboard for presentation

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
|-- app/
|   |-- streamlit_app.py
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

1. Add environment variables in `.env` when using Alpaca as your selected data source:

```env
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
```

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
2. Engineer 28 features.
3. Build EWMA volatility-normalized labels.
4. Construct month-indexed walk-forward-ready table.

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

### Step 15: Streamlit Demo

Launch dashboard:

```bash
streamlit run app/streamlit_app.py
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
