"""Command-line entrypoint for the full intraday modeling workflow."""

from __future__ import annotations

import argparse

from src.pipeline import prepare_engineered_universe, run_training_pipeline
from src.utils.config import ensure_directories, load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for common project execution modes."""
    parser = argparse.ArgumentParser(description="AAI 6640 intraday direction project runner")

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["data", "train", "full"],
        default="full",
        help="data: download+prepare only, train/full: run complete training pipeline",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download market data even if cached parquet files exist",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of models: lstm cnn_lstm tft",
    )

    return parser.parse_args()


def main() -> None:
    """Execute selected mode with clean, commented control flow."""
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)

    if args.mode == "data":
        # Data mode runs download + feature + label generation without training.
        modeling_frame, feature_columns = prepare_engineered_universe(
            config=config,
            force_refresh=args.force_refresh,
        )
        print(f"Prepared rows: {len(modeling_frame):,}")
        print(f"Feature count: {len(feature_columns)}")
        return

    # Train/full mode executes all folds and model comparisons.
    output = run_training_pipeline(
        config_path=args.config,
        model_names=args.models,
        force_refresh=args.force_refresh,
    )

    print("Training complete. Model summaries:")
    for model_name, model_summary in output["summary"]["models"].items():
        macro_f1 = model_summary["summary"]["test_macro_f1"]["mean"]
        print(f"  - {model_name}: mean test macro F1 = {macro_f1:.4f}")


if __name__ == "__main__":
    main()
