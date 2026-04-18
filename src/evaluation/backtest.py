"""Backtesting utilities with confidence filtering and risk metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def mc_dropout_predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    ticker_id: torch.Tensor,
    passes: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate predictive mean and uncertainty using MC dropout at inference."""
    model.train()  # Keep dropout active during inference passes.
    probs: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(passes):
            logits = model(x, ticker_id=ticker_id)
            probs.append(torch.softmax(logits, dim=-1))

    stacked = torch.stack(probs, dim=0)
    mean_prob = stacked.mean(dim=0)
    std_prob = stacked.std(dim=0)
    return mean_prob, std_prob


def predictions_to_position(
    predicted_class: pd.Series,
    confidence: pd.Series | None = None,
    confidence_threshold: float | None = None,
) -> pd.Series:
    """Convert model classes into long/flat positions with optional confidence gating."""
    positions: list[float] = []
    current_position = 0.0

    for idx, label in enumerate(predicted_class.to_numpy(dtype=np.int64)):
        # Optional confidence filter: only act on sufficiently certain predictions.
        if (
            confidence_threshold is not None
            and confidence is not None
            and float(confidence.iloc[idx]) < confidence_threshold
        ):
            positions.append(current_position)
            continue

        if label == 2:  # Up -> enter/keep long.
            current_position = 1.0
        elif label == 0:  # Down -> exit to cash.
            current_position = 0.0
        # Neutral keeps prior position by design.

        positions.append(current_position)

    return pd.Series(positions, index=predicted_class.index, name="position")


def compute_max_drawdown(equity_curve: pd.Series) -> tuple[float, int]:
    """Return max drawdown percentage and maximum drawdown duration in bars."""
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1.0

    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    duration = 0
    max_duration = 0
    for value in drawdown:
        if value < 0:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0

    return max_drawdown, max_duration


def compute_risk_metrics(
    strategy_returns: pd.Series,
    risk_free_rate_annual: float = 0.02,
    periods_per_year: int = 252 * 78,
) -> dict[str, float]:
    """Compute Sharpe, Sortino, Calmar, annual return, and volatility statistics."""
    returns = strategy_returns.fillna(0.0)

    if returns.empty:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
        }

    mean_return = float(returns.mean())
    std_return = float(returns.std(ddof=0))
    downside_std = float(returns.loc[returns < 0].std(ddof=0))

    annualized_return = mean_return * periods_per_year
    annualized_volatility = std_return * np.sqrt(periods_per_year)

    # If variance is effectively zero, risk-adjusted ratios are not informative.
    if std_return < 1e-10:
        return {
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
        }

    risk_free_per_period = risk_free_rate_annual / periods_per_year
    excess_return = mean_return - risk_free_per_period

    sharpe = (excess_return / max(std_return, 1e-12)) * np.sqrt(periods_per_year)
    sortino = (excess_return / max(downside_std, 1e-12)) * np.sqrt(periods_per_year)

    equity_curve = (1.0 + returns).cumprod()
    max_drawdown, _ = compute_max_drawdown(equity_curve)
    calmar = annualized_return / max(abs(max_drawdown), 1e-12)

    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
    }


def run_backtest(
    frame: pd.DataFrame,
    price_column: str = "close",
    pred_column: str = "pred_label",
    confidence_column: str | None = "pred_confidence",
    confidence_threshold: float | None = None,
    risk_free_rate_annual: float = 0.02,
) -> dict[str, Any]:
    """Simulate strategy and return equity curve plus risk/return metrics."""
    base = frame.copy()

    # If multi-ticker rows are present, keep position/returns isolated by ticker,
    # then aggregate equally by timestamp into one portfolio return stream.
    if "ticker" in base.columns:
        base = base.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        base["position"] = 0.0
        for _, ticker_idx in base.groupby("ticker", sort=False).groups.items():
            group = base.loc[ticker_idx]
            conf = group[confidence_column] if confidence_column and confidence_column in group.columns else None
            positions = predictions_to_position(
                predicted_class=group[pred_column],
                confidence=conf,
                confidence_threshold=confidence_threshold,
            )
            base.loc[ticker_idx, "position"] = positions.to_numpy(dtype=np.float64)
        base["asset_return"] = (
            base.groupby("ticker", sort=False)[price_column]
            .pct_change()
            .fillna(0.0)
        )
        base["strategy_return"] = (
            base.groupby("ticker", sort=False)["position"].shift(1).fillna(0.0)
            * base["asset_return"]
        )

        df = (
            base.groupby("timestamp", as_index=False)
            .agg(
                asset_return=("asset_return", "mean"),
                strategy_return=("strategy_return", "mean"),
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        df = base.sort_values("timestamp").reset_index(drop=True)

        confidence_series = df[confidence_column] if confidence_column and confidence_column in df.columns else None

        # Step 1: map predictions to executable position states.
        df["position"] = predictions_to_position(
            predicted_class=df[pred_column],
            confidence=confidence_series,
            confidence_threshold=confidence_threshold,
        )

        # Step 2: compute asset and strategy returns.
        df["asset_return"] = df[price_column].pct_change().fillna(0.0)
        df["strategy_return"] = df["position"].shift(1).fillna(0.0) * df["asset_return"]

    # Step 3: compute cumulative performance curves.
    df["equity_curve"] = (1.0 + df["strategy_return"]).cumprod()
    df["benchmark_curve"] = (1.0 + df["asset_return"]).cumprod()

    # Step 4: calculate risk-adjusted metrics.
    strategy_metrics = compute_risk_metrics(
        strategy_returns=df["strategy_return"],
        risk_free_rate_annual=risk_free_rate_annual,
    )
    benchmark_metrics = compute_risk_metrics(
        strategy_returns=df["asset_return"],
        risk_free_rate_annual=risk_free_rate_annual,
    )

    max_dd, max_dd_duration = compute_max_drawdown(df["equity_curve"])

    return {
        "results_frame": df,
        "strategy_metrics": {
            **strategy_metrics,
            "max_drawdown": float(max_dd),
            "max_drawdown_duration_bars": int(max_dd_duration),
            "final_equity": float(df["equity_curve"].iloc[-1]),
        },
        "benchmark_metrics": {
            **benchmark_metrics,
            "final_equity": float(df["benchmark_curve"].iloc[-1]),
        },
    }
