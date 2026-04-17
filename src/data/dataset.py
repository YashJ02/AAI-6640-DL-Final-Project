"""PyTorch dataset, windowing, and walk-forward split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.features import TECHNICAL_COLUMNS


@dataclass
class WalkForwardSplit:
    """Container for one walk-forward split specification."""

    fold_id: int
    split_column: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


class IntradaySequenceDataset(Dataset):
    """Sequence dataset that returns (features, label, ticker_id) tuples."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int,
    ) -> None:
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.frame = frame.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

        self.features = self.frame[feature_columns].to_numpy(dtype=np.float32)
        self.labels = self.frame["label"].to_numpy(dtype=np.int64)
        self.ticker_ids = self.frame["ticker_id"].to_numpy(dtype=np.int64)

        # Build valid sample endpoints so windows never cross ticker boundaries.
        self.sample_end_indices: list[int] = []
        for _, ticker_slice in self.frame.groupby("ticker", sort=False):
            start = int(ticker_slice.index.min())
            end = int(ticker_slice.index.max())
            first_valid = start + sequence_length - 1
            if first_valid <= end:
                self.sample_end_indices.extend(range(first_valid, end + 1))

    def __len__(self) -> int:
        return len(self.sample_end_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end_idx = self.sample_end_indices[index]
        start_idx = end_idx - self.sequence_length + 1

        x = self.features[start_idx : end_idx + 1]
        y = self.labels[end_idx]
        ticker_id = self.ticker_ids[end_idx]

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(ticker_id, dtype=torch.long),
        )


def assign_month_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Add month_index (1..N) relative to earliest timestamp in the dataset."""
    df = frame.copy()

    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")

    first_year = int(ts.dt.year.min())
    first_month = int(ts.loc[ts.dt.year == first_year].dt.month.min())
    base = first_year * 12 + first_month

    # Convert calendar year/month into consecutive month ids starting at 1.
    month_number = ts.dt.year * 12 + ts.dt.month
    df["month_index"] = (month_number - base + 1).astype(int)
    return df


def assign_session_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Add session_index (1..N) based on local NY trading dates."""
    df = frame.copy()

    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    session = ts.dt.date

    unique_sessions = pd.Index(sorted(session.unique()))
    session_to_id = {day: idx + 1 for idx, day in enumerate(unique_sessions)}
    df["session_index"] = session.map(session_to_id).astype(int)

    return df


def _parse_month_based_splits(config: dict[str, Any]) -> list[WalkForwardSplit]:
    """Parse explicit month-index split definitions from config."""
    splits: list[WalkForwardSplit] = []

    for idx, fold in enumerate(config["dataset"]["walk_forward_folds"], start=1):
        splits.append(
            WalkForwardSplit(
                fold_id=idx,
                split_column="month_index",
                train_start=int(fold["train_months"][0]),
                train_end=int(fold["train_months"][1]),
                val_start=int(fold["val_month"]),
                val_end=int(fold["val_month"]),
                test_start=int(fold["test_month"]),
                test_end=int(fold["test_month"]),
            )
        )

    return splits


def _parse_session_based_splits(frame: pd.DataFrame, config: dict[str, Any]) -> list[WalkForwardSplit]:
    """Generate rolling session-index folds for short-horizon intraday datasets."""
    cfg = config["dataset"]["session_split"]

    train_sessions = int(cfg["train_sessions"])
    val_sessions = int(cfg["val_sessions"])
    test_sessions = int(cfg["test_sessions"])
    step_sessions = int(cfg.get("step_sessions", test_sessions))
    max_folds = int(cfg.get("max_folds", 0))

    total_sessions = int(frame["session_index"].max())
    fold_span = train_sessions + val_sessions + test_sessions

    if total_sessions < fold_span:
        raise ValueError(
            "Not enough sessions for configured session_split. "
            f"Need at least {fold_span}, found {total_sessions}."
        )

    splits: list[WalkForwardSplit] = []
    fold_id = 1
    start_session = 1

    while True:
        train_start = start_session
        train_end = train_start + train_sessions - 1
        val_start = train_end + 1
        val_end = val_start + val_sessions - 1
        test_start = val_end + 1
        test_end = test_start + test_sessions - 1

        if test_end > total_sessions:
            break

        splits.append(
            WalkForwardSplit(
                fold_id=fold_id,
                split_column="session_index",
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        fold_id += 1
        start_session += step_sessions

        if max_folds > 0 and len(splits) >= max_folds:
            break

    if not splits:
        raise ValueError("Session split generation produced zero folds")

    return splits


def parse_walk_forward_splits(frame: pd.DataFrame, config: dict[str, Any]) -> list[WalkForwardSplit]:
    """Parse split definitions from config into strongly-typed objects."""
    split_mode = str(config["dataset"].get("split_mode", "month")).lower()

    if split_mode == "sessions":
        return _parse_session_based_splits(frame=frame, config=config)

    return _parse_month_based_splits(config=config)


def build_ticker_id_map(frame: pd.DataFrame) -> dict[str, int]:
    """Map ticker symbols to contiguous integer ids for embeddings."""
    tickers = sorted(frame["ticker"].dropna().unique().tolist())
    return {ticker: idx for idx, ticker in enumerate(tickers)}


def fit_indicator_normalization(
    train_frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Fit z-score stats on train split only for selected indicator columns."""
    columns = columns or TECHNICAL_COLUMNS

    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        mean = float(train_frame[column].mean())
        std = float(train_frame[column].std(ddof=0))
        stats[column] = {"mean": mean, "std": std if std > 1e-12 else 1.0}

    return stats


def apply_indicator_normalization(
    frame: pd.DataFrame,
    stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply pre-fitted indicator normalization statistics to any split."""
    df = frame.copy()

    for column, values in stats.items():
        if column in df.columns:
            df[column] = (df[column] - values["mean"]) / values["std"]

    return df


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for focal/cross-entropy style losses."""
    if labels.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)

    inv = 1.0 / counts
    normalized = inv * (num_classes / inv.sum())

    return torch.tensor(normalized, dtype=torch.float32)


def _split_frame_by_range(frame: pd.DataFrame, split: WalkForwardSplit) -> dict[str, pd.DataFrame]:
    """Slice dataframe into train/val/test partitions for one walk-forward fold."""
    key = split.split_column

    train_mask = (frame[key] >= split.train_start) & (frame[key] <= split.train_end)
    val_mask = (frame[key] >= split.val_start) & (frame[key] <= split.val_end)
    test_mask = (frame[key] >= split.test_start) & (frame[key] <= split.test_end)

    return {
        "train": frame.loc[train_mask].copy(),
        "val": frame.loc[val_mask].copy(),
        "test": frame.loc[test_mask].copy(),
    }


def _drop_non_finite_rows(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Remove rows that contain non-finite feature values before sequence windowing."""
    df = frame.copy()

    available_features = [column for column in feature_columns if column in df.columns]
    if available_features:
        df[available_features] = df[available_features].replace([np.inf, -np.inf], np.nan)

    return df.dropna(subset=available_features + ["label"]).reset_index(drop=True)


def create_fold_dataloaders(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    split: WalkForwardSplit,
) -> dict[str, Any]:
    """Create normalized datasets and dataloaders for a single fold."""
    splits = _split_frame_by_range(frame, split)

    # Train-only fit for normalization to avoid temporal leakage.
    normalization_stats = fit_indicator_normalization(splits["train"], TECHNICAL_COLUMNS)

    train_frame = apply_indicator_normalization(splits["train"], normalization_stats)
    val_frame = apply_indicator_normalization(splits["val"], normalization_stats)
    test_frame = apply_indicator_normalization(splits["test"], normalization_stats)

    # Guard against both NaN and +/-inf values before sequence creation.
    train_frame = _drop_non_finite_rows(train_frame, feature_columns)
    val_frame = _drop_non_finite_rows(val_frame, feature_columns)
    test_frame = _drop_non_finite_rows(test_frame, feature_columns)

    sequence_length = int(config["dataset"]["sequence_length"])
    batch_size = int(config["dataset"]["batch_size"])

    train_dataset = IntradaySequenceDataset(train_frame, feature_columns, sequence_length)
    val_dataset = IntradaySequenceDataset(val_frame, feature_columns, sequence_length)
    test_dataset = IntradaySequenceDataset(test_frame, feature_columns, sequence_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
        drop_last=False,
    )

    class_weights = compute_class_weights(train_frame["label"].to_numpy(), num_classes=3)

    return {
        "fold": split,
        "normalization_stats": normalization_stats,
        "frames": {"train": train_frame, "val": val_frame, "test": test_frame},
        "datasets": {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "class_weights": class_weights,
    }


def create_all_fold_dataloaders(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create DataLoaders for all configured walk-forward folds."""
    df = frame.copy()
    df = assign_month_index(df)
    df = assign_session_index(df)

    ticker_to_id = build_ticker_id_map(df)
    df["ticker_id"] = df["ticker"].map(ticker_to_id)

    outputs: list[dict[str, Any]] = []
    for split in parse_walk_forward_splits(frame=df, config=config):
        fold_output = create_fold_dataloaders(
            frame=df,
            feature_columns=feature_columns,
            config=config,
            split=split,
        )
        fold_output["ticker_to_id"] = ticker_to_id
        outputs.append(fold_output)

    return outputs
