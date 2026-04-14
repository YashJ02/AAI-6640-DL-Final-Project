"""Performance analysis across volatility regimes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def assign_volatility_regime(
    frame: pd.DataFrame,
    atr_column: str = "atr",
    percentile: float = 0.75,
) -> pd.DataFrame:
    """Label each bar as high-vol or low-vol using session-level ATR percentiles."""
    df = frame.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    session = ts.dt.date

    # Aggregate ATR at session level, then threshold top quartile as high volatility.
    daily_atr = df.groupby(session)[atr_column].mean()
    threshold = float(daily_atr.quantile(percentile))

    daily_regime = pd.Series(
        np.where(daily_atr >= threshold, "high_vol", "low_vol"),
        index=daily_atr.index,
    )

    df["volatility_regime"] = session.map(daily_regime)
    return df


def evaluate_predictions_by_regime(
    frame: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    regime_col: str = "volatility_regime",
) -> pd.DataFrame:
    """Compute accuracy/macro-F1 separately for each volatility regime."""
    rows: list[dict[str, Any]] = []

    for regime in ["low_vol", "high_vol"]:
        subset = frame.loc[frame[regime_col] == regime]
        if subset.empty:
            rows.append(
                {
                    "regime": regime,
                    "count": 0,
                    "accuracy": float("nan"),
                    "macro_f1": float("nan"),
                }
            )
            continue

        y_true = subset[y_true_col].to_numpy(dtype=np.int64)
        y_pred = subset[y_pred_col].to_numpy(dtype=np.int64)

        rows.append(
            {
                "regime": regime,
                "count": int(len(subset)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def compute_regime_degradation(metrics_by_regime: pd.DataFrame) -> dict[str, float]:
    """Compute percent degradation from low-vol to high-vol performance."""
    low = metrics_by_regime.loc[metrics_by_regime["regime"] == "low_vol"]
    high = metrics_by_regime.loc[metrics_by_regime["regime"] == "high_vol"]

    if low.empty or high.empty:
        return {"accuracy_degradation_pct": float("nan"), "macro_f1_degradation_pct": float("nan")}

    low_acc = float(low.iloc[0]["accuracy"])
    high_acc = float(high.iloc[0]["accuracy"])
    low_f1 = float(low.iloc[0]["macro_f1"])
    high_f1 = float(high.iloc[0]["macro_f1"])

    acc_deg = ((low_acc - high_acc) / max(low_acc, 1e-8)) * 100.0
    f1_deg = ((low_f1 - high_f1) / max(low_f1, 1e-8)) * 100.0

    return {
        "accuracy_degradation_pct": float(acc_deg),
        "macro_f1_degradation_pct": float(f1_deg),
    }
