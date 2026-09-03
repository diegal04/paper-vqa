"""Shared configuration and filesystem helpers for CLI entry points."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def resolved_config(config: DictConfig) -> dict[str, Any]:
    """Convert a fully resolved Hydra config into plain serialisable objects."""
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("root Hydra configuration must resolve to a dictionary")
    return {str(key): value for key, value in resolved.items()}


def resolve_device(value: str) -> str:
    """Resolve the user-friendly ``auto`` device setting deterministically."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def write_manifests(manifests: tuple[Any, ...], directory: Path) -> None:
    """Persist all source manifests alongside a run before training starts."""
    for manifest in manifests:
        safe_name = manifest.name.replace(":", "_").replace("/", "_")
        manifest.save(directory / "manifests" / f"{safe_name}.json")
