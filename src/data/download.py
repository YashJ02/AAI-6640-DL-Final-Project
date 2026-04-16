"""Download and cache intraday OHLCV data from yfinance."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils.config import flatten_tickers

try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def _parse_interval(interval: str) -> tuple[int, str]:
    """Parse intervals such as 5Min into quantity and unit."""
    if not interval.endswith("Min"):
        raise ValueError(f"Unsupported interval format: {interval}")

    qty = int(interval.replace("Min", ""))
    return qty, "minute"


def _to_yfinance_interval(interval: str) -> str:
    """Translate project interval notation to yfinance interval notation."""
    qty, unit = _parse_interval(interval)
    if unit != "minute":
        raise ValueError(f"Unsupported yfinance interval unit: {unit}")
    return f"{qty}m"


def _cache_file_path(cache_dir: Path, ticker: str, interval: str, history_days: int) -> Path:
    """Build deterministic cache path so reruns re-use files."""
    filename = f"{ticker}_{interval}_{history_days}d.parquet"
    return cache_dir / filename


def _canonical_column_name(column: Any) -> str:
    """Flatten provider-specific column labels into simple names."""
    if isinstance(column, tuple):
        for part in column:
            part_text = str(part).strip()
            if part_text:
                return part_text
        return str(column[0])

    text = str(column)
    if text.startswith("(") and text.endswith(")"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, tuple):
                for part in parsed:
                    part_text = str(part).strip()
                    if part_text:
                        return part_text
        except (ValueError, SyntaxError):
            pass

    return text


def _normalize_bar_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize provider-specific columns into a shared schema."""
    flattened = frame.copy()
    flattened.columns = [_canonical_column_name(col) for col in flattened.columns]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Datetime": "timestamp",
        "Date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "timestamp": "timestamp",
    }

    normalized = flattened.rename(columns=rename_map).copy()

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")

    normalized = normalized[required]
    normalized["ticker"] = ticker
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)

    return normalized


def _download_yfinance(ticker: str, interval: str, history_days: int) -> pd.DataFrame:
    """Download one ticker from yfinance."""
    if not HAS_YFINANCE:
        raise RuntimeError("yfinance is not installed")

    yf_interval = _to_yfinance_interval(interval)
    raw = yf.download(
        tickers=ticker,
        period=f"{history_days + 5}d",
        interval=yf_interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no bars for {ticker}")

    raw = raw.reset_index()
    # yfinance may expose Datetime or Date, depending on interval and exchange.
    if "Datetime" not in raw.columns and "Date" in raw.columns:
        raw = raw.rename(columns={"Date": "Datetime"})

    normalized = _normalize_bar_columns(raw, ticker=ticker)
    return normalized


def download_ticker_data(
    ticker: str,
    config: dict[str, Any],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download data for one ticker with cache-first behavior."""
    cache_dir = Path(config["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    interval = config["data"]["interval"]
    history_days = int(config["data"]["history_days"])
    cache_path = _cache_file_path(cache_dir, ticker, interval, history_days)

    # Step 1: return cached parquet when available.
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(set(cached.columns)):
            cached = _normalize_bar_columns(cached, ticker=ticker)
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached.sort_values("timestamp").reset_index(drop=True)

    # Step 2: download from open yfinance source.
    frame = _download_yfinance(ticker=ticker, interval=interval, history_days=history_days)
    frame.to_parquet(cache_path, index=False)
    return frame


def download_universe(config: dict[str, Any], force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Download all configured tickers and return a mapping ticker->dataframe."""
    all_tickers = flatten_tickers(config["tickers"])
    outputs: dict[str, pd.DataFrame] = {}

    for ticker in tqdm(all_tickers, desc="Downloading tickers"):
        outputs[ticker] = download_ticker_data(
            ticker=ticker,
            config=config,
            force_refresh=force_refresh,
        )

    return outputs
