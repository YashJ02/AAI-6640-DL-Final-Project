"""Data quality checks and cleaning utilities for intraday OHLCV bars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _parse_interval_minutes(interval: str) -> int:
    """Parse project interval notation like 5Min into integer minutes."""
    if not interval.endswith("Min"):
        raise ValueError(f"Unsupported interval format: {interval}")
    return int(interval.replace("Min", ""))


def _expected_bars_per_session(config: dict[str, Any]) -> int:
    """Compute expected intraday bars per regular session for the configured interval."""
    interval_minutes = _parse_interval_minutes(str(config["data"]["interval"]))

    session_start = int(config["data"]["start_hour"]) * 60 + int(config["data"]["start_minute"])
    session_end = int(config["data"]["end_hour"]) * 60 + int(config["data"]["end_minute"])

    if session_end <= session_start:
        raise ValueError("Session end must be greater than session start")

    duration = session_end - session_start
    # Use [start, end) semantics for bars to avoid expecting a non-existent 16:00 bar.
    return max(1, (duration + interval_minutes - 1) // interval_minutes)


def clean_ohlcv_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    ticker: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run conservative cleaning so model training uses high-quality, non-noisy bars."""
    cleaning_cfg = config.get("cleaning", {})

    enabled = bool(cleaning_cfg.get("enabled", True))
    min_session_coverage = float(cleaning_cfg.get("min_session_coverage", 0.85))
    max_abs_log_return = float(cleaning_cfg.get("max_abs_log_return", 0.20))
    max_intrabar_range = float(cleaning_cfg.get("max_intrabar_range", 0.25))
    drop_zero_volume = bool(cleaning_cfg.get("drop_zero_volume", True))

    df = frame.copy()
    before_rows = int(len(df))

    if not enabled:
        stats = {
            "ticker": ticker,
            "rows_before": before_rows,
            "rows_after": before_rows,
            "dropped_rows": 0,
            "drop_pct": 0.0,
            "dropped_duplicate_ts": 0,
            "dropped_non_session": 0,
            "dropped_invalid_ohlcv": 0,
            "dropped_outlier_return": 0,
            "dropped_outlier_range": 0,
            "dropped_low_coverage_session": 0,
            "kept_sessions": int(pd.to_datetime(df["timestamp"], utc=True).dt.date.nunique()),
        }
        return df, stats

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1) Remove duplicate timestamps.
    before_dup = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    dropped_duplicate_ts = before_dup - len(df)

    # 2) Keep only regular session bars.
    local_ts = df["timestamp"].dt.tz_convert(config["data"]["timezone"])
    minute_of_day = local_ts.dt.hour * 60 + local_ts.dt.minute
    session_start = int(config["data"]["start_hour"]) * 60 + int(config["data"]["start_minute"])
    session_end = int(config["data"]["end_hour"]) * 60 + int(config["data"]["end_minute"])

    # Keep bars in [session_start, session_end), e.g., 09:30 ... 15:55 for 5-minute bars.
    session_mask = (minute_of_day >= session_start) & (minute_of_day < session_end)
    dropped_non_session = int((~session_mask).sum())
    df = df.loc[session_mask].reset_index(drop=True)

    # 3) Remove rows with invalid OHLCV values and physically inconsistent bars.
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = (
        df["open"].gt(0.0)
        & df["high"].gt(0.0)
        & df["low"].gt(0.0)
        & df["close"].gt(0.0)
        & df["volume"].ge(0.0)
        & df[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
    )
    if drop_zero_volume:
        valid_mask &= df["volume"].gt(0.0)

    ohlc_consistent = (
        df["high"].ge(df[["open", "close", "low"]].max(axis=1))
        & df["low"].le(df[["open", "close", "high"]].min(axis=1))
    )

    combined_valid = valid_mask & ohlc_consistent
    dropped_invalid_ohlcv = int((~combined_valid).sum())
    df = df.loc[combined_valid].reset_index(drop=True)

    # 4) Remove extreme per-bar anomalies likely caused by bad ticks.
    # Compute returns within each trading session so overnight gaps are not treated as bad ticks.
    local_ts = df["timestamp"].dt.tz_convert(config["data"]["timezone"])
    session_date = local_ts.dt.date
    close_prev = df.groupby(session_date, sort=False)["close"].shift(1).replace(0.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.log(df["close"] / close_prev)

    keep_return = log_ret.abs().le(max_abs_log_return) | log_ret.isna()
    dropped_outlier_return = int((~keep_return).sum())
    df = df.loc[keep_return].reset_index(drop=True)

    intrabar_range = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
    keep_range = intrabar_range.le(max_intrabar_range) | intrabar_range.isna()
    dropped_outlier_range = int((~keep_range).sum())
    df = df.loc[keep_range].reset_index(drop=True)

    # 5) Remove sessions with low bar coverage.
    local_ts = df["timestamp"].dt.tz_convert(config["data"]["timezone"])
    session_date = local_ts.dt.date
    session_counts = df.groupby(session_date, sort=False).size()

    expected_bars = _expected_bars_per_session(config)
    min_bars = max(1, int(expected_bars * min_session_coverage))
    keep_sessions = session_counts.loc[session_counts >= min_bars].index

    session_keep_mask = session_date.isin(keep_sessions)
    dropped_low_coverage_session = int((~session_keep_mask).sum())
    df = df.loc[session_keep_mask].reset_index(drop=True)

    after_rows = int(len(df))
    dropped_rows = before_rows - after_rows

    stats = {
        "ticker": ticker,
        "rows_before": before_rows,
        "rows_after": after_rows,
        "dropped_rows": int(dropped_rows),
        "drop_pct": float((dropped_rows / max(before_rows, 1)) * 100.0),
        "dropped_duplicate_ts": int(dropped_duplicate_ts),
        "dropped_non_session": int(dropped_non_session),
        "dropped_invalid_ohlcv": int(dropped_invalid_ohlcv),
        "dropped_outlier_return": int(dropped_outlier_return),
        "dropped_outlier_range": int(dropped_outlier_range),
        "dropped_low_coverage_session": int(dropped_low_coverage_session),
        "kept_sessions": int(len(keep_sessions)),
    }

    return df, stats
