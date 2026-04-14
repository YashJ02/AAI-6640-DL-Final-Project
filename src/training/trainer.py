"""Model training loop with early stopping, checkpointing, and MLflow logging."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.models.cnn_lstm import DilatedCNNLSTMModel
from src.models.lstm import LSTMTemporalAttentionModel
from src.models.tft import TemporalFusionTransformerModel
from src.training.losses import FocalLoss
from src.training.metrics import (
    compute_classification_metrics,
    mcnemar_test,
    majority_class_baseline,
    summarize_fold_metrics,
)

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


@dataclass
class EarlyStopping:
    """Stop training when validation metric no longer improves."""

    patience: int
    mode: str = "max"
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best_value = -float("inf") if self.mode == "max" else float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        improved = False
        if self.mode == "max":
            improved = value > (self.best_value + self.min_delta)
        else:
            improved = value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def build_model(model_name: str, config: dict[str, Any]) -> nn.Module:
    """Factory that builds one of the three project architectures."""
    num_features = int(config["models"]["num_features"])
    num_classes = int(config["models"]["num_classes"])

    if model_name == "lstm":
        model_cfg = config["models"]["lstm"]
        return LSTMTemporalAttentionModel(
            num_features=num_features,
            num_classes=num_classes,
            hidden_size=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            attention_size=int(model_cfg["attention_size"]),
            fc_hidden=int(model_cfg["fc_hidden"]),
        )

    if model_name == "tft":
        model_cfg = config["models"]["tft"]
        return TemporalFusionTransformerModel(
            num_features=num_features,
            num_classes=num_classes,
            ticker_vocab_size=int(config["models"]["ticker_vocab_size"]),
            hidden_size=int(model_cfg["hidden_size"]),
            attention_heads=int(model_cfg["attention_heads"]),
            ticker_embedding_dim=int(model_cfg["ticker_embedding_dim"]),
            dropout=float(model_cfg["dropout"]),
        )

    if model_name == "cnn_lstm":
        model_cfg = config["models"]["cnn_lstm"]
        return DilatedCNNLSTMModel(
            num_features=num_features,
            num_classes=num_classes,
            conv_channels=int(model_cfg["conv_channels"]),
            conv_kernel_size=int(model_cfg["conv_kernel_size"]),
            dilations=list(model_cfg["dilations"]),
            lstm_hidden_size=int(model_cfg["lstm_hidden_size"]),
            lstm_layers=int(model_cfg["lstm_layers"]),
            dropout=float(model_cfg["dropout"]),
            fc_hidden=int(model_cfg["fc_hidden"]),
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def _warmup_cosine_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay multiplier."""
    if total_steps <= 0:
        return 1.0

    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))

    progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(cosine)


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, Any]:
    """Run one train/eval epoch and return metrics plus raw predictions."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_samples = 0

    all_true: list[int] = []
    all_pred: list[int] = []

    for x, y, ticker_id in loader:
        x = x.to(device)
        y = y.to(device)
        ticker_id = ticker_id.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x, ticker_id=ticker_id)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=1)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    if total_samples == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "classification_report": {},
            "y_true": np.array([], dtype=np.int64),
            "y_pred": np.array([], dtype=np.int64),
        }

    y_true_arr = np.array(all_true, dtype=np.int64)
    y_pred_arr = np.array(all_pred, dtype=np.int64)

    metrics = compute_classification_metrics(y_true=y_true_arr, y_pred=y_pred_arr)
    metrics["loss"] = total_loss / total_samples
    metrics["y_true"] = y_true_arr
    metrics["y_pred"] = y_pred_arr

    return metrics


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    path: Path,
) -> None:
    """Persist best model state so fold evaluation can reload exact weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": metric,
        },
        path,
    )


def _load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    """Restore model weights from checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def train_one_fold(
    model_name: str,
    fold_bundle: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train one model on one walk-forward fold and evaluate on test split."""
    fold = fold_bundle["fold"]
    train_loader = fold_bundle["loaders"]["train"]
    val_loader = fold_bundle["loaders"]["val"]
    test_loader = fold_bundle["loaders"]["test"]

    model = build_model(model_name, config).to(device)

    class_weights = fold_bundle["class_weights"].to(device)
    criterion = FocalLoss(
        gamma=float(config["training"]["focal_loss"]["gamma"]),
        alpha=class_weights,
        label_smoothing=float(config["training"]["label_smoothing"]),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    max_epochs = int(config["training"]["max_epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = int(total_steps * float(config["training"]["warmup_ratio"]))

    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: _warmup_cosine_lambda(step, warmup_steps, total_steps),
    )

    early_stopping = EarlyStopping(
        patience=int(config["training"]["early_stopping_patience"]),
        mode="max",
    )

    checkpoint_path = (
        Path(config["training"]["checkpoint_dir"])
        / f"{model_name}_fold_{fold.fold_id}.pt"
    )

    history: list[dict[str, float]] = []
    best_val_f1 = -float("inf")
    best_epoch = 0

    run_context = mlflow.start_run(run_name=f"{model_name}_fold_{fold.fold_id}") if HAS_MLFLOW else nullcontext()

    with run_context:
        if HAS_MLFLOW:
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "fold_id": fold.fold_id,
                    "learning_rate": config["training"]["learning_rate"],
                    "weight_decay": config["training"]["weight_decay"],
                    "max_epochs": max_epochs,
                    "batch_size": config["dataset"]["batch_size"],
                    "label_smoothing": config["training"]["label_smoothing"],
                    "focal_gamma": config["training"]["focal_loss"]["gamma"],
                }
            )

        for epoch in tqdm(range(1, max_epochs + 1), desc=f"{model_name} fold {fold.fold_id}"):
            train_metrics = _run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip_norm=float(config["training"]["gradient_clip_norm"]),
            )

            val_metrics = _run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            epoch_record = {
                "epoch": float(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_loss": float(val_metrics["loss"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
            history.append(epoch_record)

            if HAS_MLFLOW:
                mlflow.log_metrics(
                    {
                        "train_loss": epoch_record["train_loss"],
                        "train_macro_f1": epoch_record["train_macro_f1"],
                        "val_loss": epoch_record["val_loss"],
                        "val_macro_f1": epoch_record["val_macro_f1"],
                    },
                    step=epoch,
                )

            # Keep checkpoint with best validation macro F1.
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                _save_checkpoint(model, optimizer, epoch, best_val_f1, checkpoint_path)

            if early_stopping.step(float(val_metrics["macro_f1"])):
                break

        if not checkpoint_path.exists():
            _save_checkpoint(model, optimizer, epoch=max_epochs, metric=float("nan"), path=checkpoint_path)

        # Evaluate test using best checkpoint.
        _load_checkpoint(model, checkpoint_path, device)
        test_metrics = _run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        if HAS_MLFLOW:
            mlflow.log_metrics(
                {
                    "test_loss": float(test_metrics["loss"]),
                    "test_accuracy": float(test_metrics["accuracy"]),
                    "test_macro_f1": float(test_metrics["macro_f1"]),
                }
            )
            mlflow.log_artifact(str(checkpoint_path))

    train_labels = fold_bundle["frames"]["train"]["label"].to_numpy(dtype=np.int64)
    baseline = majority_class_baseline(
        y_true=test_metrics["y_true"],
        y_train=train_labels,
    )

    result = {
        "model_name": model_name,
        "fold_id": fold.fold_id,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "test_classification_report": test_metrics["classification_report"],
        "baseline_accuracy": baseline["accuracy"],
        "baseline_macro_f1": baseline["macro_f1"],
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "y_true": test_metrics["y_true"],
        "y_pred": test_metrics["y_pred"],
    }

    return result


def train_model_across_folds(
    model_name: str,
    fold_bundles: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Train one model architecture across all walk-forward folds."""
    if HAS_MLFLOW:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_names"][model_name])

    fold_results: list[dict[str, Any]] = []
    for fold_bundle in fold_bundles:
        fold_result = train_one_fold(
            model_name=model_name,
            fold_bundle=fold_bundle,
            config=config,
            device=device,
        )
        fold_results.append(fold_result)

    summary_input = [
        {
            "test_loss": result["test_loss"],
            "test_accuracy": result["test_accuracy"],
            "test_macro_f1": result["test_macro_f1"],
            "baseline_accuracy": result["baseline_accuracy"],
            "baseline_macro_f1": result["baseline_macro_f1"],
        }
        for result in fold_results
    ]

    return {
        "model_name": model_name,
        "fold_results": fold_results,
        "summary": summarize_fold_metrics(summary_input),
    }


def pairwise_mcnemar_across_models(
    model_outputs: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, float]]]:
    """Run pairwise McNemar tests for each fold across model outputs."""
    results: dict[str, list[dict[str, float]]] = {}

    for model_a, model_b in combinations(model_outputs.keys(), 2):
        key = f"{model_a}_vs_{model_b}"
        fold_stats: list[dict[str, float]] = []

        folds_a = model_outputs[model_a]["fold_results"]
        folds_b = model_outputs[model_b]["fold_results"]

        for res_a, res_b in zip(folds_a, folds_b, strict=False):
            stat = mcnemar_test(
                y_true=res_a["y_true"],
                y_pred_a=res_a["y_pred"],
                y_pred_b=res_b["y_pred"],
            )
            stat["fold_id"] = float(res_a["fold_id"])
            fold_stats.append(stat)

        results[key] = fold_stats

    return results
