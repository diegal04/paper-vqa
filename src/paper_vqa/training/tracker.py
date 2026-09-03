"""Optional Weights & Biases tracking behind a narrow typed interface."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    """Minimal event interface used by the trainer."""

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics at a global step."""

    def finish(self) -> None:
        """Finish the run and flush remote events."""

    def log_artifact(self, path: Path, name: str, artifact_type: str) -> None:
        """Store a file or directory as a reproducibility artefact."""


class NullTracker:
    """No-op tracker for deterministic offline unit tests and local debugging."""

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Discard metrics intentionally."""

    def finish(self) -> None:
        """Perform no cleanup."""

    def log_artifact(self, path: Path, name: str, artifact_type: str) -> None:
        """Discard local artefact references intentionally."""


class WandbTracker:
    """W&B adapter that keeps the SDK out of training-domain code."""

    def __init__(self, config: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> None:
        """Start the configured W&B run.

        Args:
            config: Logging configuration.
            resolved_config: Full Hydra configuration stored with the run.
        """
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=config["project"],
            entity=config.get("entity") or None,
            mode=config["mode"],
            group=config.get("group") or None,
            tags=list(config.get("tags", [])),
            config=dict(resolved_config),
        )

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics through the active W&B run."""
        self._run.log(dict(metrics), step=step)

    def finish(self) -> None:
        """Finish the W&B run."""
        self._run.finish()

    def log_artifact(self, path: Path, name: str, artifact_type: str) -> None:
        """Upload a file or directory while retaining its run-local copy."""
        if not path.exists():
            return
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact)


def build_tracker(config: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> Tracker:
    """Return a W&B tracker only when the Hydra config enables it."""
    return WandbTracker(config, resolved_config) if bool(config["enabled"]) else NullTracker()
