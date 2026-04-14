"""Orchestration helpers for feature importance, volatility, and backtesting steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.evaluation.backtest import run_backtest
from src.evaluation.feature_importance import (
    extract_tft_vsn_importance,
    lstm_ablation_importance,
    merge_importance_rankings,
    mutual_information_ranking,
)
from src.evaluation.volatility_analysis import (
    assign_volatility_regime,
    compute_regime_degradation,
    evaluate_predictions_by_regime,
)


def run_feature_importance_pipeline(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
    tft_model: torch.nn.Module | None = None,
    tft_loader: torch.utils.data.DataLoader | None = None,
    device: torch.device | None = None,
    baseline_lstm_f1: float | None = None,
    ablation_callback: Any | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute MI, optional TFT VSN, and optional LSTM ablation rankings."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mi_df = mutual_information_ranking(
        frame=frame,
        feature_columns=feature_columns,
        label_column="label",
    )

    # Use placeholder tables when optional components are unavailable.
    if tft_model is not None and tft_loader is not None and device is not None:
        vsn_df = extract_tft_vsn_importance(
            model=tft_model,
            loader=tft_loader,
            feature_columns=feature_columns,
            device=device,
        )
    else:
        vsn_df = pd.DataFrame({"feature": feature_columns, "vsn_weight": 0.0, "rank": 0})

    if baseline_lstm_f1 is not None and ablation_callback is not None:
        ablation_df = lstm_ablation_importance(
            feature_columns=feature_columns,
            baseline_macro_f1=baseline_lstm_f1,
            train_eval_callback=ablation_callback,
        )
    else:
        ablation_df = pd.DataFrame({"feature": feature_columns, "f1_drop": 0.0, "rank": 0})

    combined_df = merge_importance_rankings(mi_df, vsn_df, ablation_df)

    mi_df.to_csv(out_path / "mi_ranking.csv", index=False)
    vsn_df.to_csv(out_path / "vsn_ranking.csv", index=False)
    ablation_df.to_csv(out_path / "ablation_ranking.csv", index=False)
    combined_df.to_csv(out_path / "combined_feature_importance.csv", index=False)

    return {
        "mi": mi_df,
        "vsn": vsn_df,
        "ablation": ablation_df,
        "combined": combined_df,
    }


def run_volatility_and_backtest_pipeline(
    predictions_frame: pd.DataFrame,
    output_dir: str | Path,
    confidence_threshold: float = 0.6,
    risk_free_rate_annual: float = 0.02,
) -> dict[str, Any]:
    """Run volatility-regime analysis and strategy backtest from prediction table."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: assign volatility regimes from ATR and score per regime.
    regime_frame = assign_volatility_regime(predictions_frame, atr_column="atr", percentile=0.75)
    regime_metrics = evaluate_predictions_by_regime(
        frame=regime_frame,
        y_true_col="label",
        y_pred_col="pred_label",
        regime_col="volatility_regime",
    )
    degradation = compute_regime_degradation(regime_metrics)

    # Step 2: run unfiltered and confidence-filtered backtests.
    backtest_unfiltered = run_backtest(
        frame=predictions_frame,
        pred_column="pred_label",
        confidence_column="pred_confidence",
        confidence_threshold=None,
        risk_free_rate_annual=risk_free_rate_annual,
    )
    backtest_filtered = run_backtest(
        frame=predictions_frame,
        pred_column="pred_label",
        confidence_column="pred_confidence",
        confidence_threshold=confidence_threshold,
        risk_free_rate_annual=risk_free_rate_annual,
    )

    regime_metrics.to_csv(out_path / "volatility_regime_metrics.csv", index=False)
    backtest_unfiltered["results_frame"].to_csv(out_path / "backtest_unfiltered_curve.csv", index=False)
    backtest_filtered["results_frame"].to_csv(out_path / "backtest_filtered_curve.csv", index=False)

    payload = {
        "degradation": degradation,
        "backtest_unfiltered": {
            "strategy_metrics": backtest_unfiltered["strategy_metrics"],
            "benchmark_metrics": backtest_unfiltered["benchmark_metrics"],
        },
        "backtest_filtered": {
            "strategy_metrics": backtest_filtered["strategy_metrics"],
            "benchmark_metrics": backtest_filtered["benchmark_metrics"],
        },
    }

    with (out_path / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "regime_metrics": regime_metrics,
        "degradation": degradation,
        "backtest_unfiltered": backtest_unfiltered,
        "backtest_filtered": backtest_filtered,
    }
