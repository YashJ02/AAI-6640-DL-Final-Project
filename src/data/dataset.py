"""PyTorch dataset, windowing, and walk-forward split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
    train_values: list[int] | None = None
    val_values: list[int] | None = None
    test_values: list[int] | None = None


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


def _parse_kfold_based_splits(frame: pd.DataFrame, config: dict[str, Any]) -> list[WalkForwardSplit]:
    """Generate K-fold splits over session ids with optional time-aware leakage guard."""
    cfg = config["dataset"].get("k_fold", {})

    n_splits = int(cfg.get("n_splits", 5))
    val_fraction = float(cfg.get("val_fraction", 0.2))
    time_aware = bool(cfg.get("time_aware", True))
    max_folds = int(cfg.get("max_folds", 0))

    if n_splits < 3:
        raise ValueError("k_fold.n_splits must be at least 3")

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("k_fold.val_fraction must be in (0, 1)")

    sessions = np.array(sorted(frame["session_index"].unique()), dtype=np.int64)
    if sessions.size < n_splits:
        raise ValueError(
            "Not enough sessions for configured k_fold. "
            f"Need at least {n_splits}, found {sessions.size}."
        )

    chunks = [chunk.tolist() for chunk in np.array_split(sessions, n_splits) if len(chunk) > 0]

    splits: list[WalkForwardSplit] = []
    fold_id = 1

    for test_idx, test_chunk in enumerate(chunks):
        test_sessions = [int(value) for value in test_chunk]

        if time_aware:
            train_val_pool = [
                int(value)
                for previous_chunk in chunks[:test_idx]
                for value in previous_chunk
            ]
        else:
            train_val_pool = [
                int(value)
                for idx, chunk in enumerate(chunks)
                if idx != test_idx
                for value in chunk
            ]

        # Need at least one session each for train and val.
        if len(train_val_pool) < 2:
            continue

        val_size = max(1, int(round(len(train_val_pool) * val_fraction)))
        val_size = min(val_size, len(train_val_pool) - 1)

        train_sessions = train_val_pool[:-val_size]
        val_sessions = train_val_pool[-val_size:]

        if not train_sessions or not val_sessions or not test_sessions:
            continue

        splits.append(
            WalkForwardSplit(
                fold_id=fold_id,
                split_column="session_index",
                train_start=0,
                train_end=0,
                val_start=0,
                val_end=0,
                test_start=0,
                test_end=0,
                train_values=train_sessions,
                val_values=val_sessions,
                test_values=test_sessions,
            )
        )

        fold_id += 1
        if max_folds > 0 and len(splits) >= max_folds:
            break

    if not splits:
        raise ValueError(
            "K-fold split generation produced zero folds. "
            "If time_aware=true, try increasing sessions or reducing n_splits."
        )

    return splits


def parse_walk_forward_splits(frame: pd.DataFrame, config: dict[str, Any]) -> list[WalkForwardSplit]:
    """Parse split definitions from config into strongly-typed objects."""
    split_mode = str(config["dataset"].get("split_mode", "month")).lower()

    if split_mode == "sessions":
        return _parse_session_based_splits(frame=frame, config=config)

    if split_mode == "kfold":
        return _parse_kfold_based_splits(frame=frame, config=config)

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


def _apply_fold_adaptive_labels(
    splits: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Optionally recalibrate thresholds per fold using train-only normalized returns."""
    adaptive_cfg = config.get("labels", {}).get("per_fold_adaptive_thresholds", {})
    enabled = bool(adaptive_cfg.get("enabled", False))

    train_df = splits["train"]
    if not enabled or "normalized_return" not in train_df.columns:
        return splits

    train_norm = train_df["normalized_return"].replace([np.inf, -np.inf], np.nan).dropna()
    if train_norm.empty:
        return splits

    down_q = float(adaptive_cfg.get("down_quantile", 0.33))
    up_q = float(adaptive_cfg.get("up_quantile", 0.67))

    threshold_down = float(train_norm.quantile(down_q))
    threshold_up = float(train_norm.quantile(up_q))

    if not np.isfinite(threshold_down) or not np.isfinite(threshold_up) or threshold_down >= threshold_up:
        threshold_down = float(config["labels"]["threshold_down"])
        threshold_up = float(config["labels"]["threshold_up"])

    recalibrated: dict[str, pd.DataFrame] = {}
    for split_name, split_frame in splits.items():
        df = split_frame.copy()
        if "normalized_return" in df.columns:
            z = df["normalized_return"].to_numpy(dtype=np.float64)
            labels = np.full(shape=len(df), fill_value=1, dtype=np.int64)
            labels[z < threshold_down] = 0
            labels[z > threshold_up] = 2
            df["label"] = labels
        recalibrated[split_name] = df

    return recalibrated


def _build_weighted_train_sampler(
    dataset: IntradaySequenceDataset,
    num_classes: int = 3,
) -> WeightedRandomSampler | None:
    """Create weighted sampler from sequence-end labels to balance train batches."""
    if len(dataset) == 0:
        return None

    sample_ends = np.asarray(dataset.sample_end_indices, dtype=np.int64)
    sample_labels = dataset.labels[sample_ends]

    counts = np.bincount(sample_labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv[sample_labels]

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def _split_frame_by_range(frame: pd.DataFrame, split: WalkForwardSplit) -> dict[str, pd.DataFrame]:
    """Slice dataframe into train/val/test partitions for one walk-forward fold."""
    key = split.split_column

    if split.train_values is not None:
        train_mask = frame[key].isin(split.train_values)
    else:
        train_mask = (frame[key] >= split.train_start) & (frame[key] <= split.train_end)

    if split.val_values is not None:
        val_mask = frame[key].isin(split.val_values)
    else:
        val_mask = (frame[key] >= split.val_start) & (frame[key] <= split.val_end)

    if split.test_values is not None:
        test_mask = frame[key].isin(split.test_values)
    else:
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
    splits = _apply_fold_adaptive_labels(splits, config)

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

    num_workers = int(config["dataset"].get("num_workers", 0))
    pin_memory = bool(config["dataset"].get("pin_memory", False))
    persistent_workers = bool(config["dataset"].get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(config["dataset"].get("prefetch_factor", 2))
    weighted_sampler_enabled = bool(config["dataset"].get("weighted_sampler", False))

    common_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = persistent_workers
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_sampler = _build_weighted_train_sampler(train_dataset) if weighted_sampler_enabled else None
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_kwargs,
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
