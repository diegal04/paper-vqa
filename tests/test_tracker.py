import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paper_vqa.training.tracker import ArtifactPolicy


class FakeRun:
    """Minimal W&B run double used to inspect adapter calls."""

    def __init__(self) -> None:
        """Initialise captured log and artefact events."""
        self.logged: list[tuple[dict[str, float], int]] = []
        self.artifacts: list[FakeArtifact] = []
        self.finished = False

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Capture one scalar log event."""
        self.logged.append((metrics, step))

    def log_artifact(self, artifact: "FakeArtifact") -> None:
        """Capture one allowed artefact."""
        self.artifacts.append(artifact)

    def finish(self) -> None:
        """Record that the adapter closed the run."""
        self.finished = True


class FakeArtifact:
    """Minimal W&B artefact double."""

    def __init__(self, name: str, type: str) -> None:
        """Store the logical artefact identity."""
        self.name = name
        self.type = type
        self.paths: list[tuple[Path, str | None]] = []

    def add_file(self, path: str, name: str | None = None) -> None:
        """Capture a file addition."""
        self.paths.append((Path(path), name))


def test_artifact_policy_uses_conservative_defaults() -> None:
    """Weights and predictions must require explicit upload consent."""
    policy = ArtifactPolicy.from_mapping({})

    assert policy.permits("dataset")
    assert not policy.permits("model")
    assert not policy.permits("evaluation")
    assert not policy.permits("unknown")


def test_artifact_policy_honours_explicit_flags() -> None:
    """Every supported artefact category follows its independent switch."""
    policy = ArtifactPolicy.from_mapping(
        {
            "upload_manifests": False,
            "upload_checkpoints": True,
            "upload_predictions": True,
        }
    )

    assert not policy.permits("dataset")
    assert policy.permits("model")
    assert policy.permits("evaluation")


def test_wandb_adapter_names_run_and_filters_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK adapter should use stable names and block heavy artefacts by default."""
    from paper_vqa.training.tracker import WandbTracker

    run = FakeRun()
    init_arguments: dict[str, Any] = {}

    def fake_init(**kwargs: Any) -> FakeRun:
        """Capture W&B initialisation arguments and return the test run."""
        init_arguments.update(kwargs)
        return run

    fake_wandb = SimpleNamespace(init=fake_init, Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = {
        "project": "paper-vqa",
        "entity": None,
        "mode": "offline",
        "group": "setup",
        "tags": [],
        "directory": str(tmp_path),
        "upload_manifests": True,
        "upload_checkpoints": False,
        "upload_predictions": False,
    }
    tracker = WandbTracker(
        config,
        {"experiment": {"name": "convergence"}, "trainer": {"seed": 42}},
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    manifest_directory = tmp_path / "manifests"
    manifest_directory.mkdir()
    nested_manifest = manifest_directory / "nested" / "validation.json"
    nested_manifest.parent.mkdir()
    nested_manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"weights")

    tracker.log({"val/loss": 1.0}, step=3)
    tracker.log_artifact(manifest_directory, "data-manifests", "dataset")
    tracker.log_artifact(checkpoint, "selected-checkpoint", "model")
    tracker.finish()

    assert init_arguments["name"] == "convergence-seed_42"
    assert init_arguments["dir"] == str(tmp_path)
    assert run.logged == [({"val/loss": 1.0}, 3)]
    assert [artifact.type for artifact in run.artifacts] == ["dataset"]
    assert run.artifacts[0].paths == [(nested_manifest, "nested/validation.json")]
    assert run.finished
