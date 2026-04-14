"""Download and cache intraday OHLCV data."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.config import flatten_tickers

try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    HAS_ALPACA = True
except ImportError:
    HAS_ALPACA = False

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


def _normalize_bar_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize provider-specific columns into a shared schema."""
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Datetime": "timestamp",
        "timestamp": "timestamp",
    }

    normalized = frame.rename(columns=rename_map).copy()

    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns for {ticker}: {missing}")

    normalized = normalized[required]
    normalized["ticker"] = ticker
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized = normalized.sort_values("timestamp").reset_index(drop=True)

    return normalized


def _download_alpaca(
    ticker: str,
    interval: str,
    history_days: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Download one ticker from Alpaca using API credentials from .env."""
    if not HAS_ALPACA:
        raise RuntimeError("alpaca-py is not installed")

    load_dotenv()
    key_id = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")

    if not key_id or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET in environment")

    qty, unit = _parse_interval(interval)
    if unit != "minute":
        raise ValueError(f"Unsupported Alpaca timeframe unit: {unit}")

    start_dt = datetime.utcnow() - timedelta(days=history_days + 5)
    end_dt = datetime.utcnow()

    client = StockHistoricalDataClient(api_key=key_id, secret_key=secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame(amount=qty, unit=TimeFrameUnit.Minute),
        start=start_dt,
        end=end_dt,
        adjustment="raw",
        feed="iex",
    )

    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise RuntimeError(f"Alpaca returned no bars for {ticker}")

    # Alpaca returns a MultiIndex; flatten into regular columns.
    bars = bars.reset_index()
    bars = bars.rename(columns={"symbol": "ticker", "timestamp": "timestamp"})

    normalized = _normalize_bar_columns(bars, ticker=ticker)

    # Keep only regular market session to match feature definitions.
    ny_timestamps = normalized["timestamp"].dt.tz_convert(config["data"]["timezone"])
    session_mask = (
        ((ny_timestamps.dt.hour > config["data"]["start_hour"]) |
         ((ny_timestamps.dt.hour == config["data"]["start_hour"]) &
          (ny_timestamps.dt.minute >= config["data"]["start_minute"])))
        &
        ((ny_timestamps.dt.hour < config["data"]["end_hour"]) |
         ((ny_timestamps.dt.hour == config["data"]["end_hour"]) &
          (ny_timestamps.dt.minute == config["data"]["end_minute"])))
    )

    normalized = normalized.loc[session_mask].reset_index(drop=True)
    return normalized


def _download_yfinance(ticker: str, interval: str, history_days: int) -> pd.DataFrame:
    """Download one ticker from yfinance as fallback."""
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
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached.sort_values("timestamp").reset_index(drop=True)

    source_primary = config["data"]["source_primary"]
    source_fallback = config["data"]["source_fallback"]

    # Step 2: try primary data source first.
    if source_primary == "alpaca":
        try:
            frame = _download_alpaca(ticker=ticker, interval=interval, history_days=history_days, config=config)
            frame.to_parquet(cache_path, index=False)
            return frame
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Alpaca download failed for {ticker}: {exc}")

    # Step 3: fallback to yfinance if primary failed.
    if source_fallback == "yfinance":
        frame = _download_yfinance(ticker=ticker, interval=interval, history_days=history_days)
        frame.to_parquet(cache_path, index=False)
        return frame

    raise RuntimeError(f"Unable to download data for {ticker} from configured sources")


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
