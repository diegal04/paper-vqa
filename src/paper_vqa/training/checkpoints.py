"""Safe, self-describing model checkpoint artefacts."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_model, save_model
from torch import nn


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Non-tensor information accompanying a safe model checkpoint."""

    epoch: int
    monitor_name: str
    monitor_value: float
    threshold: float | None
    config: dict[str, Any]


class CheckpointManager:
    """Save model weights in safetensors format with JSON metadata."""

    def __init__(self, directory: Path) -> None:
        """Create the output directory lazily and retain its path."""
        self.directory = directory

    @property
    def weights_path(self) -> Path:
        """Return the deterministic best-weight filename."""
        return self.directory / "model.safetensors"

    @property
    def metadata_path(self) -> Path:
        """Return the deterministic metadata filename."""
        return self.directory / "metadata.json"

    def save(self, model: nn.Module, metadata: CheckpointMetadata) -> None:
        """Persist model state and structured run metadata without pickle."""
        self.directory.mkdir(parents=True, exist_ok=True)
        save_model(model, str(self.weights_path))
        self.metadata_path.write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True, default=str), encoding="utf-8"
        )

    def load(self, model: nn.Module) -> CheckpointMetadata:
        """Load safe weights and return the accompanying metadata."""
        load_model(model, str(self.weights_path), strict=True)
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return CheckpointMetadata(**raw)
