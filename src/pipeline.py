"""End-to-end experiment orchestration from data download to model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.dataset import create_all_fold_dataloaders
from src.data.cleaning import clean_ohlcv_frame
from src.data.download import download_related_universe, download_universe
from src.data.features import engineer_features
from src.data.labels import build_labels
from src.training.trainer import pairwise_mcnemar_across_models, train_model_across_folds
from src.utils.config import ensure_directories, load_config, resolve_device, set_seed


def _to_builtin(value: Any) -> Any:
    """Convert NumPy/Torch types into JSON-serializable native Python types."""
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _symbol_feature_prefix(symbol: str) -> str:
    """Build a stable feature prefix from a related symbol string."""
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(symbol))
    token = token.strip("_")
    return token or "symbol"


def _build_related_feature_frame(
    related_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame | None, list[str]]:
    """Convert related-symbol OHLCV frames into aligned context feature columns."""
    if not related_frames:
        return None, []

    merged: pd.DataFrame | None = None
    related_columns: list[str] = []

    for symbol, frame in related_frames.items():
        prefix = f"rel_{_symbol_feature_prefix(symbol)}"
        columns = [f"{prefix}_ret1", f"{prefix}_volchg1"]
        related_columns.extend(columns)

        rel = frame[["timestamp", "close", "volume"]].copy().sort_values("timestamp").reset_index(drop=True)

        close_prev = rel["close"].shift(1).replace(0.0, np.nan)
        volume_prev = rel["volume"].shift(1).replace(0.0, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel[columns[0]] = np.log(rel["close"] / close_prev)
            rel[columns[1]] = np.log(rel["volume"] / volume_prev)

        rel[columns] = rel[columns].replace([np.inf, -np.inf], np.nan)

        rel = rel[["timestamp"] + columns]
        rel = rel.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        if merged is None:
            merged = rel
        else:
            merged = merged.merge(rel, on="timestamp", how="outer")

    if merged is None:
        return None, []

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged[related_columns] = merged[related_columns].replace([np.inf, -np.inf], np.nan)
    merged[related_columns] = merged[related_columns].ffill().fillna(0.0)
    return merged, related_columns


def _write_data_quality_outputs(
    cleaning_rows: list[dict[str, Any]],
    per_ticker_rows: list[dict[str, Any]],
    combined: pd.DataFrame,
    feature_columns: list[str],
    related_feature_count: int,
) -> None:
    """Persist data-quality and dataset-understanding artifacts for traceability."""
    quality_dir = Path("artifacts") / "data_quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(cleaning_rows).to_csv(quality_dir / "ticker_cleaning_report.csv", index=False)
    pd.DataFrame(per_ticker_rows).to_csv(quality_dir / "ticker_modeling_report.csv", index=False)

    label_distribution = (
        combined["label"].value_counts(normalize=True).sort_index().to_dict()
        if "label" in combined.columns
        else {}
    )

    summary = {
        "rows": int(len(combined)),
        "tickers": int(combined["ticker"].nunique()) if "ticker" in combined.columns else 0,
        "feature_count": int(len(feature_columns)),
        "related_feature_count": int(related_feature_count),
        "timestamp_min": str(combined["timestamp"].min()) if "timestamp" in combined.columns else None,
        "timestamp_max": str(combined["timestamp"].max()) if "timestamp" in combined.columns else None,
        "label_distribution": label_distribution,
    }

    with (quality_dir / "modeling_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(summary), handle, indent=2)


def prepare_engineered_universe(
    config: dict[str, Any],
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Download and transform all tickers into one labeled modeling dataframe."""
    raw_frames = download_universe(config=config, force_refresh=force_refresh)
    related_frames = (
        download_related_universe(config=config, force_refresh=force_refresh)
        if bool(config.get("data", {}).get("use_related_features", False))
        else {}
    )
    related_feature_frame, related_feature_columns = _build_related_feature_frame(related_frames)

    engineered_frames: list[pd.DataFrame] = []
    base_feature_columns: list[str] | None = None

    cleaning_rows: list[dict[str, Any]] = []
    per_ticker_rows: list[dict[str, Any]] = []

    for ticker, raw_frame in raw_frames.items():
        cleaned_frame, cleaning_stats = clean_ohlcv_frame(raw_frame, config=config, ticker=ticker)
        cleaning_rows.append(cleaning_stats)

        if cleaned_frame.empty:
            continue

        # Step 1: engineer stationary/technical/temporal features.
        feature_frame, base_feature_columns, _ = engineer_features(
            frame=cleaned_frame,
            config=config,
            fit_normalizer=False,
            normalizer_stats=None,
        )

        if related_feature_frame is not None and related_feature_columns:
            feature_frame = feature_frame.merge(related_feature_frame, on="timestamp", how="left")
            feature_frame[related_feature_columns] = feature_frame[related_feature_columns].replace(
                [np.inf, -np.inf],
                np.nan,
            )
            feature_frame[related_feature_columns] = feature_frame[related_feature_columns].ffill().fillna(0.0)

        # Step 2: generate volatility-normalized labels.
        labeled_frame, _ = build_labels(feature_frame, config=config)
        labeled_frame["ticker"] = ticker

        label_counts = labeled_frame["label"].value_counts(normalize=True).to_dict()
        per_ticker_rows.append(
            {
                "ticker": ticker,
                "rows_after_cleaning": int(len(cleaned_frame)),
                "rows_after_features": int(len(feature_frame)),
                "rows_after_labels": int(len(labeled_frame)),
                "label_0_pct": float(label_counts.get(0, 0.0)),
                "label_1_pct": float(label_counts.get(1, 0.0)),
                "label_2_pct": float(label_counts.get(2, 0.0)),
            }
        )

        engineered_frames.append(labeled_frame)

    if not engineered_frames or base_feature_columns is None:
        raise RuntimeError("No engineered data produced from configured tickers")

    feature_columns = base_feature_columns + related_feature_columns

    combined = pd.concat(engineered_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    _write_data_quality_outputs(
        cleaning_rows=cleaning_rows,
        per_ticker_rows=per_ticker_rows,
        combined=combined,
        feature_columns=feature_columns,
        related_feature_count=len(related_feature_columns),
    )

    return combined, feature_columns


def _strip_fold_for_summary(fold_result: dict[str, Any]) -> dict[str, Any]:
    """Keep concise fold metrics for report JSON while dropping large arrays."""
    keep_keys = [
        "model_name",
        "fold_id",
        "best_epoch",
        "best_val_macro_f1",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
        "test_confusion_matrix",
        "baseline_accuracy",
        "baseline_macro_f1",
        "checkpoint_path",
    ]
    return {key: fold_result[key] for key in keep_keys}


def run_training_pipeline(
    config_path: str = "config/default.yaml",
    model_names: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run full training workflow across all folds and model architectures."""
    config = load_config(config_path)
    ensure_directories(config)
    set_seed(int(config["experiment"]["random_seed"]))
    device = resolve_device(config["experiment"]["device"])

    # Step 1: build the supervised modeling table.
    modeling_frame, feature_columns = prepare_engineered_universe(
        config=config,
        force_refresh=force_refresh,
    )

    # Keep model dimensions synchronized with active feature and ticker universe.
    config["models"]["num_features"] = int(len(feature_columns))
    config["models"]["ticker_vocab_size"] = int(modeling_frame["ticker"].nunique())

    # Step 2: build fold-specific datasets/loaders.
    fold_bundles = create_all_fold_dataloaders(
        frame=modeling_frame,
        feature_columns=feature_columns,
        config=config,
    )

    selected_models = model_names or ["lstm", "cnn_lstm", "tft"]
    model_outputs: dict[str, dict[str, Any]] = {}

    # Step 3: train each architecture across walk-forward folds.
    for model_name in selected_models:
        model_outputs[model_name] = train_model_across_folds(
            model_name=model_name,
            fold_bundles=fold_bundles,
            config=config,
            device=device,
        )

    # Step 4: run pairwise significance tests.
    mcnemar = pairwise_mcnemar_across_models(model_outputs)

    summary = {
        "device": str(device),
        "feature_columns": feature_columns,
        "models": {
            name: {
                "summary": output["summary"],
                "folds": [_strip_fold_for_summary(item) for item in output["fold_results"]],
            }
            for name, output in model_outputs.items()
        },
        "mcnemar": mcnemar,
    }

    # Step 5: persist concise summary and per-fold prediction arrays.
    artifacts_dir = Path("artifacts")
    predictions_dir = artifacts_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for model_name, output in model_outputs.items():
        for fold_result in output["fold_results"]:
            np.savez_compressed(
                predictions_dir / f"{model_name}_fold_{fold_result['fold_id']}.npz",
                y_true=fold_result["y_true"],
                y_pred=fold_result["y_pred"],
            )

    with (artifacts_dir / "results_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_to_builtin(summary), handle, indent=2)

    return {
        "summary": summary,
        "raw_model_outputs": model_outputs,
        "modeling_frame": modeling_frame,
        "feature_columns": feature_columns,
    }
