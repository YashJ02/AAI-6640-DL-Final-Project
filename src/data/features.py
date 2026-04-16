"""Feature engineering for stationary OHLCV, technical indicators, and temporal features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

STATIONARY_COLUMNS = [
    "log_return",
    "hl_range",
    "oc_body",
    "upper_shadow",
    "volume_log_change",
]

TECHNICAL_COLUMNS = [
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "ema_9",
    "ema_21",
    "atr",
    "vwap",
    "obv",
    "adx",
    "stoch_k",
    "stoch_d",
    "cci",
    "williams_r",
    "mfi",
]

TEMPORAL_COLUMNS = [
    "tod_sin_primary",
    "tod_cos_primary",
    "tod_sin_harmonic",
    "tod_cos_harmonic",
    "dow_norm",
]


def _nan_series(index: pd.Index) -> pd.Series:
    """Create an aligned NaN series for cases where an indicator is unavailable."""
    return pd.Series(np.nan, index=index, dtype=float)


def _to_series_or_nan(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Convert indicator output to aligned series, defaulting to NaN when missing."""
    if values is None:
        return _nan_series(index)

    series = pd.Series(values)
    return series.reindex(index).astype(float)


def _pick_column(frame: pd.DataFrame | None, prefix: str, index: pd.Index) -> pd.Series:
    """Return first indicator column matching prefix or NaN series when unavailable."""
    if frame is None or frame.empty:
        return _nan_series(index)

    candidates = [col for col in frame.columns if col.startswith(prefix)]
    if not candidates:
        return _nan_series(index)

    return _to_series_or_nan(frame[candidates[0]], index)


def add_stationary_ohlcv_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Step 1: replace non-stationary OHLCV levels with return-based representations."""
    df = frame.copy()

    # Log return is additive and largely scale-invariant across price levels.
    close_prev = df["close"].shift(1).replace(0.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_return"] = np.log(df["close"] / close_prev)

    # High-low range captures intrabar volatility as a fraction of price.
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)

    # Open-close body captures directional move during the bar.
    df["oc_body"] = (df["close"] - df["open"]) / df["close"].replace(0.0, np.nan)

    # Upper shadow captures rejection after upward movement.
    df["upper_shadow"] = (
        df["high"] - np.maximum(df["open"], df["close"])
    ) / df["close"].replace(0.0, np.nan)

    # Log volume change makes volume more stationary across sessions.
    volume_prev = df["volume"].shift(1).replace(0.0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["volume_log_change"] = np.log(df["volume"] / volume_prev)

    # Replace divide-by-zero infinities so warm-up cleanup can safely drop them.
    df[STATIONARY_COLUMNS] = df[STATIONARY_COLUMNS].replace([np.inf, -np.inf], np.nan)

    return df


def _compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute cumulative intraday VWAP with daily reset."""
    local_ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    session_date = local_ts.dt.date

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cumulative_tp_vol = (typical_price * df["volume"]).groupby(session_date).cumsum()
    cumulative_vol = df["volume"].groupby(session_date).cumsum().replace(0.0, np.nan)
    return cumulative_tp_vol / cumulative_vol


def add_technical_indicators(frame: pd.DataFrame, technical_cfg: dict[str, Any]) -> pd.DataFrame:
    """Step 2: add 18 technical indicators with pandas_ta defaults from config."""
    if not HAS_PANDAS_TA:
        raise ImportError("pandas_ta is required for technical indicator computation")

    df = frame.copy()

    # Momentum indicator.
    df["rsi"] = ta.rsi(df["close"], length=technical_cfg["rsi_period"])

    # MACD trend decomposition into line, signal, and histogram.
    macd = ta.macd(
        close=df["close"],
        fast=technical_cfg["macd_fast"],
        slow=technical_cfg["macd_slow"],
        signal=technical_cfg["macd_signal"],
    )
    df["macd_line"] = _pick_column(macd, "MACD_", df.index)
    df["macd_signal"] = _pick_column(macd, "MACDs_", df.index)
    df["macd_hist"] = _pick_column(macd, "MACDh_", df.index)

    # Bollinger bands and width for volatility regime cues.
    bbands = ta.bbands(
        close=df["close"],
        length=technical_cfg["bb_period"],
        std=technical_cfg["bb_std"],
    )
    df["bb_upper"] = _pick_column(bbands, "BBU_", df.index)
    df["bb_lower"] = _pick_column(bbands, "BBL_", df.index)
    df["bb_width"] = _pick_column(bbands, "BBB_", df.index)

    # Exponential moving averages at short and medium horizons.
    df["ema_9"] = _to_series_or_nan(ta.ema(df["close"], length=technical_cfg["ema_fast"]), df.index)
    df["ema_21"] = _to_series_or_nan(ta.ema(df["close"], length=technical_cfg["ema_slow"]), df.index)

    # ATR captures absolute volatility in price units.
    df["atr"] = _to_series_or_nan(
        ta.atr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=technical_cfg["atr_period"],
        ),
        df.index,
    )

    # VWAP anchored by trading session.
    df["vwap"] = _compute_intraday_vwap(df)

    # Volume trend and trend-strength signals.
    df["obv"] = _to_series_or_nan(ta.obv(close=df["close"], volume=df["volume"]), df.index)

    adx = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=technical_cfg["adx_period"],
    )
    df["adx"] = _pick_column(adx, "ADX_", df.index)

    stoch = ta.stoch(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        k=technical_cfg["stoch_k"],
        d=technical_cfg["stoch_d"],
        smooth_k=3,
    )
    df["stoch_k"] = _pick_column(stoch, "STOCHk_", df.index)
    df["stoch_d"] = _pick_column(stoch, "STOCHd_", df.index)

    df["cci"] = _to_series_or_nan(
        ta.cci(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=technical_cfg["cci_period"],
        ),
        df.index,
    )
    df["williams_r"] = _to_series_or_nan(
        ta.willr(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            length=technical_cfg["willr_period"],
        ),
        df.index,
    )
    df["mfi"] = _to_series_or_nan(
        ta.mfi(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            volume=df["volume"],
            length=technical_cfg["mfi_period"],
        ),
        df.index,
    )

    return df


def add_fourier_temporal_features(frame: pd.DataFrame, total_minutes: int = 390) -> pd.DataFrame:
    """Step 3: add cyclical time features for intraday and weekday seasonality."""
    df = frame.copy()

    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    minutes_since_open = (ts.dt.hour * 60 + ts.dt.minute) - (9 * 60 + 30)

    # Primary cycle captures open/close effects across a full session.
    df["tod_sin_primary"] = np.sin(2 * np.pi * minutes_since_open / total_minutes)
    df["tod_cos_primary"] = np.cos(2 * np.pi * minutes_since_open / total_minutes)

    # Harmonic cycle captures secondary patterns like midday slowdowns.
    df["tod_sin_harmonic"] = np.sin(4 * np.pi * minutes_since_open / total_minutes)
    df["tod_cos_harmonic"] = np.cos(4 * np.pi * minutes_since_open / total_minutes)

    # Monday=0 through Friday=4 normalized to [0, 1].
    df["dow_norm"] = ts.dt.dayofweek / 4.0

    return df


def feature_columns() -> list[str]:
    """Return the exact ordered feature set used by all models."""
    return STATIONARY_COLUMNS + TECHNICAL_COLUMNS + TEMPORAL_COLUMNS


def fit_zscore_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Fit train-only z-score stats for selected feature columns."""
    stats: dict[str, dict[str, float]] = {}

    for column in columns:
        mean = float(frame[column].mean())
        std = float(frame[column].std(ddof=0))
        stats[column] = {"mean": mean, "std": std if std > 1e-12 else 1.0}

    return stats


def apply_zscore(frame: pd.DataFrame, stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Apply pre-fitted z-score normalization stats to a dataframe copy."""
    df = frame.copy()

    for column, values in stats.items():
        df[column] = (df[column] - values["mean"]) / values["std"]

    return df


def engineer_features(
    frame: pd.DataFrame,
    config: dict[str, Any],
    fit_normalizer: bool = False,
    normalizer_stats: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, float]] | None]:
    """Run the full feature pipeline in plan order and return engineered frame."""
    df = frame.copy().sort_values("timestamp").reset_index(drop=True)

    # Step 1: stationary OHLCV transformations.
    if config["features"]["include_stationary_ohlcv"]:
        df = add_stationary_ohlcv_features(df)

    # Step 2: technical indicator block.
    if config["features"]["include_technical_indicators"]:
        df = add_technical_indicators(df, config["features"]["technical"])

    # Step 3: Fourier temporal block.
    if config["features"]["include_fourier_time"]:
        df = add_fourier_temporal_features(df)

    all_features = feature_columns()

    # Step 4: fit/apply train-only normalization on indicator subset.
    stats_out = normalizer_stats
    if fit_normalizer:
        stats_out = fit_zscore_stats(df, TECHNICAL_COLUMNS)

    if stats_out is not None:
        df = apply_zscore(df, stats_out)

    # Step 5: drop indicator warm-up NaNs created by rolling windows.
    if config["features"]["drop_warmup_nans"]:
        df = df.dropna(subset=all_features).reset_index(drop=True)

    return df, all_features, stats_out
