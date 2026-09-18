"""Optional Weights & Biases tracking behind a narrow typed interface."""

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    """Explicit allow-list for potentially large or sensitive W&B artefacts."""

    upload_manifests: bool = True
    upload_checkpoints: bool = False
    upload_predictions: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ArtifactPolicy":
        """Build an upload policy from the logging configuration.

        Args:
            config: Resolved Hydra logging settings.

        Returns:
            Policy whose conservative defaults never upload weights or predictions.
        """
        return cls(
            upload_manifests=bool(config.get("upload_manifests", True)),
            upload_checkpoints=bool(config.get("upload_checkpoints", False)),
            upload_predictions=bool(config.get("upload_predictions", False)),
        )

    def permits(self, artifact_type: str) -> bool:
        """Return whether an artefact category may leave the local machine.

        Args:
            artifact_type: Logical type supplied by the caller.

        Returns:
            Whether the corresponding explicit configuration flag is enabled.
        """
        permissions = {
            "dataset": self.upload_manifests,
            "model": self.upload_checkpoints,
            "evaluation": self.upload_predictions,
        }
        return permissions.get(artifact_type, False)


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
        self._artifact_policy = ArtifactPolicy.from_mapping(config)
        experiment = resolved_config.get("experiment", {})
        trainer = resolved_config.get("trainer", {})
        experiment_name = str(
            experiment.get("name", "experiment")
            if isinstance(experiment, Mapping)
            else "experiment"
        )
        seed = str(trainer.get("seed", "unknown") if isinstance(trainer, Mapping) else "unknown")
        self._run_name = f"{experiment_name}-seed_{seed}"
        self._run = wandb.init(
            project=config["project"],
            entity=config.get("entity") or None,
            mode=config["mode"],
            group=config.get("group") or None,
            tags=list(config.get("tags", [])),
            name=self._run_name,
            dir=str(config["directory"]),
            config=dict(resolved_config),
        )

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        """Log scalar metrics through the active W&B run."""
        self._run.log(dict(metrics), step=step)

    def finish(self) -> None:
        """Finish the W&B run."""
        self._run.finish()

    def log_artifact(self, path: Path, name: str, artifact_type: str) -> None:
        """Upload an explicitly allowed file or directory.

        Checkpoints and per-example predictions are disabled by default because
        they are large and may contain dataset-derived content. Directory files
        are added individually because ``wandb.Artifact.add_dir`` creates a
        thread pool internally, which is unsuitable for constrained servers.
        """
        if not path.exists() or not self._artifact_policy.permits(artifact_type):
            return
        artifact = self._wandb.Artifact(name=f"{self._run_name}-{name}", type=artifact_type)
        if path.is_dir():
            files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
            for file_path in files:
                artifact.add_file(str(file_path), name=file_path.relative_to(path).as_posix())
        else:
            artifact.add_file(str(path))
        self._run.log_artifact(artifact)


def build_tracker(config: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> Tracker:
    """Return a W&B tracker only when the Hydra config enables it."""
    return WandbTracker(config, resolved_config) if bool(config["enabled"]) else NullTracker()
