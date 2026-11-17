"""File I/O: .env, datasets, LoRA adapters and reports.

PyTorch is only imported inside the adapter weight functions, so datasets and
adapter settings can be read without it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import core

ADAPTER_WEIGHTS = "adapter.pt"
ADAPTER_CONFIG = "adapter_config.json"


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def read_dataset(path: str):
    """Read a .jsonl or .csv dataset. Returns (examples, problems)."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    return core.parse_dataset_text(file.read_text(encoding="utf-8-sig"), file.suffix)


def write_jsonl(path: str, examples: list) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples) + "\n", encoding="utf-8")


def write_text(path: str, text: str) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def safe_name(name: str) -> str:
    """Make a name safe to use as a folder name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    if not cleaned:
        raise ValueError("enter a name made of letters, numbers, dashes or underscores")
    return cleaned


def list_adapters(adapters_dir: str) -> list[str]:
    folder = Path(adapters_dir)
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if (p / ADAPTER_CONFIG).is_file())


def read_adapter_config(directory: str) -> dict:
    config_file = Path(directory) / ADAPTER_CONFIG
    if not config_file.exists():
        raise FileNotFoundError(f"no adapter at {directory} - train one with `python -m app.cli train`")
    return json.loads(config_file.read_text(encoding="utf-8"))


def adapter_stamp(directory: str) -> float:
    """Last-modified time of an adapter's weights; changes after retraining (0 for none)."""
    if not directory:
        return 0.0
    weights = Path(directory) / ADAPTER_WEIGHTS
    return weights.stat().st_mtime if weights.exists() else 0.0


def save_adapter(directory: str, weights: dict, config: dict) -> None:
    """Save only the small LoRA matrices (plus the settings needed to rebuild them)."""
    import torch

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": weights}, folder / ADAPTER_WEIGHTS)
    (folder / ADAPTER_CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_adapter(directory: str):
    """Return (weights, config) for a saved adapter."""
    config = read_adapter_config(directory)
    weights_file = Path(directory) / ADAPTER_WEIGHTS
    if not weights_file.exists():
        raise FileNotFoundError(f"adapter weights missing: {weights_file}")
    import torch

    weights = torch.load(weights_file, map_location="cpu", weights_only=True)["weights"]
    return weights, config
