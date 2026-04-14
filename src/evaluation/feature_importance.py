"""Feature importance analysis: MI, TFT VSN, and LSTM ablation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif


def mutual_information_ranking(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features by non-linear mutual information with the target class."""
    if frame.empty:
        return pd.DataFrame(columns=["feature", "mi_score", "rank"])

    x = frame[feature_columns].fillna(0.0).to_numpy(dtype=np.float64)
    y = frame[label_column].to_numpy(dtype=np.int64)

    scores = mutual_info_classif(
        X=x,
        y=y,
        discrete_features=False,
        random_state=random_state,
    )

    result = pd.DataFrame({"feature": feature_columns, "mi_score": scores})
    result = result.sort_values("mi_score", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def extract_tft_vsn_importance(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    feature_columns: list[str],
    device: torch.device,
) -> pd.DataFrame:
    """Extract average variable selection weights from a trained TFT model."""
    model.eval()
    all_weights: list[np.ndarray] = []

    with torch.no_grad():
        for x, _, ticker_id in loader:
            x = x.to(device)
            ticker_id = ticker_id.to(device)

            # TFT forward returns (logits, mean_vsn_weights) when requested.
            _, vsn_weights = model(x, ticker_id=ticker_id, return_vsn_weights=True)
            all_weights.append(vsn_weights.detach().cpu().numpy())

    if not all_weights:
        return pd.DataFrame(columns=["feature", "vsn_weight", "rank"])

    stacked = np.concatenate(all_weights, axis=0)
    mean_weights = stacked.mean(axis=0)

    result = pd.DataFrame({"feature": feature_columns, "vsn_weight": mean_weights})
    result = result.sort_values("vsn_weight", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def lstm_ablation_importance(
    feature_columns: list[str],
    baseline_macro_f1: float,
    train_eval_callback: Callable[[list[str]], float],
) -> pd.DataFrame:
    """Estimate importance via leave-one-feature-out retraining."""
    rows: list[dict[str, Any]] = []

    for feature in feature_columns:
        # Build ablation set by removing one feature at a time.
        ablated_features = [name for name in feature_columns if name != feature]
        ablated_f1 = train_eval_callback(ablated_features)
        f1_drop = baseline_macro_f1 - ablated_f1

        rows.append(
            {
                "feature": feature,
                "ablated_macro_f1": float(ablated_f1),
                "f1_drop": float(f1_drop),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("f1_drop", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def merge_importance_rankings(
    mi_ranking: pd.DataFrame,
    vsn_ranking: pd.DataFrame,
    ablation_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all three ranking views into one comparison table."""
    merged = mi_ranking[["feature", "mi_score", "rank"]].rename(
        columns={"rank": "mi_rank"}
    )

    merged = merged.merge(
        vsn_ranking[["feature", "vsn_weight", "rank"]].rename(columns={"rank": "vsn_rank"}),
        on="feature",
        how="outer",
    )
    merged = merged.merge(
        ablation_ranking[["feature", "f1_drop", "rank"]].rename(columns={"rank": "ablation_rank"}),
        on="feature",
        how="outer",
    )

    return merged.sort_values("feature").reset_index(drop=True)
