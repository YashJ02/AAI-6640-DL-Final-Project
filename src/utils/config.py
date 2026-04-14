"""Configuration and reproducibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration from disk into a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    return config


def flatten_tickers(ticker_config: dict[str, list[str]]) -> list[str]:
    """Flatten sector->tickers mapping while preserving order and uniqueness."""
    seen: set[str] = set()
    ordered: list[str] = []

    for _, symbols in ticker_config.items():
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)

    return ordered


def ensure_directories(config: dict[str, Any]) -> None:
    """Create expected output directories so pipelines can write artifacts safely."""
    data_cache = Path(config["data"]["cache_dir"])
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])

    data_cache.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for deterministic behavior where possible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN flags reduce nondeterminism in recurrent/conv kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_setting: str = "auto") -> torch.device:
    """Resolve the configured device setting into an explicit torch.device."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_setting)
