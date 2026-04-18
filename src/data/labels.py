"""Volatility-normalized adaptive label generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_log_returns(
    close: pd.Series,
    horizon: int = 1,
    timestamps: pd.Series | None = None,
    timezone: str = "America/New_York",
) -> pd.Series:
    """Compute forward log returns while dropping targets that cross session boundaries."""
    returns = np.log(close.shift(-horizon) / close)

    if timestamps is None:
        return returns

    ts = pd.to_datetime(timestamps, utc=True)
    future_ts = ts.shift(-horizon)

    local_date = ts.dt.tz_convert(timezone).dt.date
    future_local_date = future_ts.dt.tz_convert(timezone).dt.date
    same_session = local_date == future_local_date

    return returns.where(same_session)


def compute_ewma_variance(
    returns: pd.Series,
    decay: float = 0.94,
    initial_variance: float | None = None,
) -> tuple[pd.Series, float]:
    """Compute EWMA variance recursively and return final variance state."""
    if returns.empty:
        raise ValueError("Returns series is empty")

    values = returns.to_numpy(dtype=float)
    variances = np.zeros_like(values)

    # Initialize with sample variance so the first few values are numerically stable.
    if initial_variance is None:
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            initial_variance = 1e-8
        else:
            initial_variance = float(np.var(finite_values[: min(50, finite_values.size)]))
        if initial_variance <= 1e-12:
            initial_variance = 1e-8

    prev_var = initial_variance
    prev_return = 0.0
    for index, value in enumerate(values):
        # RiskMetrics recursion uses previous realized return, not the current target return.
        current_var = decay * prev_var + (1.0 - decay) * (prev_return**2)
        variances[index] = current_var
        prev_var = current_var
        prev_return = float(value) if np.isfinite(value) else 0.0

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
    df["future_log_return"] = compute_log_returns(
        close=df["close"],
        horizon=horizon,
        timestamps=df["timestamp"],
        timezone=str(config["data"]["timezone"]),
    )

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
