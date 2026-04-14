"""Streamlit dashboard for intraday model comparison and analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ARTIFACTS_DIR = Path("artifacts")
SUMMARY_PATH = ARTIFACTS_DIR / "results_summary.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "analysis" / "combined_feature_importance.csv"
EVALUATION_SUMMARY_PATH = ARTIFACTS_DIR / "analysis" / "evaluation_summary.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "demo_predictions.csv"


@st.cache_data
def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact safely for dashboard rendering."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame | None:
    """Load CSV artifact safely for dashboard rendering."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def apply_custom_theme() -> None:
    """Inject custom CSS so dashboard has a distinct visual identity."""
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

          html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
          }

          .stApp {
            background: radial-gradient(circle at top left, #f6f5ec 0%, #e8efe8 45%, #dce8f1 100%);
          }

          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
          }

          .headline-card {
            border-radius: 16px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(0, 0, 0, 0.08);
            box-shadow: 0 10px 25px rgba(19, 41, 56, 0.08);
          }

          .section-title {
            margin-top: 12px;
            font-weight: 700;
            letter-spacing: 0.2px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def model_summary_table(summary: dict[str, Any]) -> pd.DataFrame:
    """Convert summary JSON into a compact table for model comparison."""
    rows: list[dict[str, Any]] = []

    for model_name, payload in summary["models"].items():
        metrics = payload["summary"]
        rows.append(
            {
                "model": model_name,
                "test_macro_f1_mean": metrics["test_macro_f1"]["mean"],
                "test_macro_f1_std": metrics["test_macro_f1"]["std"],
                "test_accuracy_mean": metrics["test_accuracy"]["mean"],
                "test_accuracy_std": metrics["test_accuracy"]["std"],
                "baseline_macro_f1_mean": metrics["baseline_macro_f1"]["mean"],
            }
        )

    return pd.DataFrame(rows).sort_values("test_macro_f1_mean", ascending=False)


def average_confusion_matrix(summary: dict[str, Any], model_name: str) -> np.ndarray:
    """Average fold-level confusion matrices for one selected model."""
    matrices = [np.array(fold["test_confusion_matrix"], dtype=float) for fold in summary["models"][model_name]["folds"]]
    if not matrices:
        return np.zeros((3, 3), dtype=float)
    return np.mean(np.stack(matrices, axis=0), axis=0)


def main() -> None:
    """Render complete project dashboard from saved artifacts."""
    st.set_page_config(page_title="Intraday Direction Dashboard", layout="wide")
    apply_custom_theme()

    summary = load_json(SUMMARY_PATH)
    feature_importance = load_csv(FEATURE_IMPORTANCE_PATH)
    evaluation_summary = load_json(EVALUATION_SUMMARY_PATH)
    predictions = load_csv(PREDICTIONS_PATH)

    st.markdown("<div class='headline-card'>", unsafe_allow_html=True)
    st.title("Intraday Stock Direction: LSTM vs TFT vs CNN-LSTM")
    st.caption("AAI 6640 Final Project Dashboard")
    st.markdown("</div>", unsafe_allow_html=True)

    if summary is None:
        st.warning(
            "No training artifacts found yet. Run `python -m src.main --mode full` to generate results."
        )
        st.stop()

    # Sidebar controls drive all dynamic views.
    available_models = list(summary["models"].keys())
    selected_model = st.sidebar.selectbox("Model", options=available_models, index=0)

    tickers = (
        sorted(predictions["ticker"].dropna().unique().tolist())
        if predictions is not None and "ticker" in predictions.columns
        else ["AAPL"]
    )
    selected_ticker = st.sidebar.selectbox("Ticker", options=tickers, index=0)

    st.markdown("### Model Comparison")
    comparison_df = model_summary_table(summary)

    left, right = st.columns([1.1, 1.4])
    with left:
        st.dataframe(comparison_df, use_container_width=True)

    with right:
        fig_bar = px.bar(
            comparison_df,
            x="model",
            y="test_macro_f1_mean",
            error_y="test_macro_f1_std",
            color="model",
            title="Mean Macro F1 Across Walk-Forward Folds",
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Confusion Matrix")
    cm = average_confusion_matrix(summary, selected_model)
    cm_fig = px.imshow(
        cm,
        x=["Pred Down", "Pred Neutral", "Pred Up"],
        y=["True Down", "True Neutral", "True Up"],
        text_auto=True,
        color_continuous_scale="YlGnBu",
        title=f"Average Confusion Matrix: {selected_model}",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("### Feature Importance")
    if feature_importance is None or feature_importance.empty:
        st.info("Feature importance artifact not found. Run evaluation pipeline to generate it.")
    else:
        top_features = feature_importance.sort_values("mi_rank").head(15)
        fi_fig = px.bar(
            top_features,
            x="feature",
            y="mi_score",
            title="Top Mutual Information Features",
            color="mi_score",
            color_continuous_scale="Tealgrn",
        )
        fi_fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown("### Live Prediction View")
    if predictions is None or predictions.empty:
        st.info("Prediction timeline file not found at artifacts/demo_predictions.csv")
    else:
        # Filter by selected model/ticker if those columns are present.
        subset = predictions.copy()
        if "model" in subset.columns:
            subset = subset.loc[subset["model"] == selected_model]
        if "ticker" in subset.columns:
            subset = subset.loc[subset["ticker"] == selected_ticker]

        if subset.empty:
            st.info("No rows for current model/ticker selection.")
        else:
            if "timestamp" in subset.columns:
                subset["timestamp"] = pd.to_datetime(subset["timestamp"])

            price_fig = go.Figure()
            price_fig.add_trace(
                go.Scatter(
                    x=subset["timestamp"] if "timestamp" in subset.columns else subset.index,
                    y=subset["close"],
                    mode="lines",
                    name="Close",
                    line={"width": 2},
                )
            )

            if "pred_label" in subset.columns:
                marker_colors = subset["pred_label"].map({0: "#d1495b", 1: "#edb458", 2: "#2a9d8f"})
                price_fig.add_trace(
                    go.Scatter(
                        x=subset["timestamp"] if "timestamp" in subset.columns else subset.index,
                        y=subset["close"],
                        mode="markers",
                        name="Predicted Class",
                        marker={"color": marker_colors, "size": 7, "opacity": 0.65},
                    )
                )

            price_fig.update_layout(title=f"{selected_ticker} Price With Prediction Overlay")
            st.plotly_chart(price_fig, use_container_width=True)

    st.markdown("### Confidence Analysis")
    if predictions is not None and "pred_confidence" in predictions.columns:
        conf_subset = predictions.copy()
        if "model" in conf_subset.columns:
            conf_subset = conf_subset.loc[conf_subset["model"] == selected_model]

        conf_fig = px.histogram(
            conf_subset,
            x="pred_confidence",
            nbins=30,
            title=f"Prediction Confidence Distribution: {selected_model}",
        )
        st.plotly_chart(conf_fig, use_container_width=True)
    else:
        st.info("Confidence column not found in prediction artifact.")

    st.markdown("### Backtesting Results")
    if evaluation_summary is None:
        st.info("No backtest summary found. Run evaluation pipeline to generate it.")
    else:
        b1, b2 = st.columns(2)
        with b1:
            st.subheader("Unfiltered")
            st.json(evaluation_summary.get("backtest_unfiltered", {}))
        with b2:
            st.subheader("Confidence Filtered")
            st.json(evaluation_summary.get("backtest_filtered", {}))


if __name__ == "__main__":
    main()
