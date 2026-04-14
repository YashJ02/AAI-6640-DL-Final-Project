"""Volatility-normalized adaptive label generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Compute log returns for the configured forecast horizon."""
    return np.log(close.shift(-horizon) / close)


def compute_ewma_variance(
    returns: pd.Series,
    decay: float = 0.94,
    initial_variance: float | None = None,
) -> tuple[pd.Series, float]:
    """Compute EWMA variance recursively and return final variance state."""
    if returns.empty:
        raise ValueError("Returns series is empty")

    values = returns.fillna(0.0).to_numpy(dtype=float)
    variances = np.zeros_like(values)

    # Initialize with sample variance so the first few values are numerically stable.
    if initial_variance is None:
        initial_variance = float(np.var(values[: min(50, len(values))]))
        if initial_variance <= 1e-12:
            initial_variance = 1e-8

    prev_var = initial_variance
    for index, value in enumerate(values):
        # RiskMetrics recursion: sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2.
        current_var = decay * prev_var + (1.0 - decay) * (value**2)
        variances[index] = current_var
        prev_var = current_var

    variance_series = pd.Series(variances, index=returns.index, name="ewma_variance")
    return variance_series, float(prev_var)


def generate_direction_labels(
    z_returns: pd.Series,
    threshold_down: float,
    threshold_up: float,
) -> pd.Series:
    """Map normalized returns into 3 classes: down (0), neutral (1), up (2)."""
    labels = np.full(shape=len(z_returns), fill_value=1, dtype=np.int64)

    labels[z_returns.to_numpy() < threshold_down] = 0
    labels[z_returns.to_numpy() > threshold_up] = 2

    return pd.Series(labels, index=z_returns.index, name="label")


def build_labels(
    frame: pd.DataFrame,
    config: dict[str, Any],
    initial_variance: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Run the complete label pipeline and return dataframe plus final EWMA state."""
    df = frame.copy().sort_values("timestamp").reset_index(drop=True)

    # Step 1: future log return over horizon (default = one bar ahead).
    horizon = int(config["dataset"]["forecast_horizon"])
    df["future_log_return"] = compute_log_returns(df["close"], horizon=horizon)

    # Step 2: EWMA variance/volatility with optional carried state.
    decay = float(config["labels"]["ewma_lambda"])
    ewma_var, final_variance = compute_ewma_variance(
        returns=df["future_log_return"],
        decay=decay,
        initial_variance=initial_variance,
    )
    df["ewma_sigma"] = np.sqrt(ewma_var.clip(lower=1e-12))

    # Step 3: normalize returns by volatility to get z-score-like targets.
    df["normalized_return"] = df["future_log_return"] / df["ewma_sigma"]

    # Step 4: fixed thresholds produce down/neutral/up classes.
    threshold_down = float(config["labels"]["threshold_down"])
    threshold_up = float(config["labels"]["threshold_up"])
    df["label"] = generate_direction_labels(
        z_returns=df["normalized_return"],
        threshold_down=threshold_down,
        threshold_up=threshold_up,
    )

    # Last rows with NaN future return are not valid supervised targets.
    df = df.dropna(subset=["future_log_return", "normalized_return", "label"]).reset_index(drop=True)

    return df, final_variance
