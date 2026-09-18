"""Safe, self-describing model checkpoint artefacts."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from safetensors.torch import load_file, load_model, save_file
from torch import Tensor, nn

WeightsScope = Literal["full", "trainable"]


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Non-tensor information accompanying a safe model checkpoint."""

    epoch: int
    monitor_name: str
    monitor_value: float
    threshold: float | None
    config: dict[str, Any]
    format_version: int = 1
    weights_scope: WeightsScope = "full"


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
        """Persist trainable weights and structured metadata without pickle.

        Frozen BLIP parameters are reproducibly reconstructed from the model
        name and immutable revision stored in the resolved configuration. New
        checkpoints therefore retain only LoRA and auxiliary-head parameters.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        trainable = _trainable_state(model)
        if not trainable:
            raise ValueError("checkpoint requires at least one trainable parameter")
        save_file(trainable, str(self.weights_path))
        stored_metadata = replace(metadata, format_version=2, weights_scope="trainable")
        self.metadata_path.write_text(
            json.dumps(asdict(stored_metadata), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def load(self, model: nn.Module) -> CheckpointMetadata:
        """Load safe weights and return the accompanying metadata."""
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata = CheckpointMetadata(**raw)
        if metadata.weights_scope == "full":
            load_model(model, str(self.weights_path), strict=True)
            return metadata
        if metadata.weights_scope != "trainable":
            raise ValueError(f"unsupported checkpoint weights scope: {metadata.weights_scope}")
        state = load_file(str(self.weights_path))
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unexpected = sorted(set(state) - expected)
            raise RuntimeError(
                "trainable checkpoint does not match model architecture; "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.load_state_dict(state, strict=False)
        return metadata


def _trainable_state(model: nn.Module) -> dict[str, Tensor]:
    """Return detached contiguous CPU tensors for parameters updated by training."""
    state = model.state_dict()
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: cast(Tensor, state[name]).detach().cpu().contiguous() for name in sorted(names)}
