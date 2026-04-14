"""End-to-end experiment orchestration from data download to model comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.dataset import create_all_fold_dataloaders
from src.data.download import download_universe
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


def prepare_engineered_universe(
    config: dict[str, Any],
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Download and transform all tickers into one labeled modeling dataframe."""
    raw_frames = download_universe(config=config, force_refresh=force_refresh)

    engineered_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    for ticker, raw_frame in raw_frames.items():
        # Step 1: engineer stationary/technical/temporal features.
        feature_frame, feature_columns, _ = engineer_features(
            frame=raw_frame,
            config=config,
            fit_normalizer=False,
            normalizer_stats=None,
        )

        # Step 2: generate volatility-normalized labels.
        labeled_frame, _ = build_labels(feature_frame, config=config)
        labeled_frame["ticker"] = ticker

        engineered_frames.append(labeled_frame)

    if not engineered_frames or feature_columns is None:
        raise RuntimeError("No engineered data produced from configured tickers")

    combined = pd.concat(engineered_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

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
