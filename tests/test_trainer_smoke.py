import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from paper_vqa.models.selective_vqa import VQAForwardOutput
from paper_vqa.training.objective import MultitaskObjective
from paper_vqa.training.tracker import NullTracker
from paper_vqa.training.trainer import Trainer, build_optimizer, resolve_warmup_steps


class TinyDataset(Dataset[dict[str, Tensor]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        label = index % 2
        return {
            "pixel_values": torch.ones(3, 2, 2),
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([0, 1]),
            "answerable": torch.tensor(label),
            "has_answerable": torch.tensor(True),
        }


class TinyVQA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.zeros(3))
        self.answerability_bias = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> VQAForwardOutput:
        batch_size, sequence_length = input_ids.shape
        logits = self.logit_bias.expand(batch_size, sequence_length, -1)
        answerability_logits = self.answerability_bias.expand(batch_size, -1)
        return VQAForwardOutput(
            logits,
            None,
            answerability_logits,
            torch.softmax(answerability_logits, dim=-1),
        )


class CapturingTracker(NullTracker):
    """No-op tracker that retains scalar events for assertions."""

    def __init__(self) -> None:
        """Initialise an empty event list."""
        self.events: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Capture one training event."""
        self.events.append((dict(metrics), step))


def test_trainer_smoke_saves_safe_checkpoint(tmp_path: Path) -> None:
    model = TinyVQA()
    tracker = CapturingTracker()
    config = {
        "device": "cpu",
        "output_dir": str(tmp_path / "checkpoint"),
        "checkpoint_monitor": "val/loss",
        "checkpoint_mode": "min",
        "epochs": 1,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
    }
    trainer = Trainer(
        model,  # type: ignore[arg-type]
        MultitaskObjective(1.0, "all_examples"),
        build_optimizer(model, config),
        None,
        tracker,
        config,
        {"smoke": True},
    )
    loader = DataLoader(TinyDataset(), batch_size=2)

    result = trainer.fit(loader, loader)

    assert "val/loss" in result
    assert (tmp_path / "checkpoint" / "model.safetensors").exists()
    assert (tmp_path / "checkpoint" / "metadata.json").exists()
    metadata = json.loads((tmp_path / "checkpoint" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["format_version"] == 2
    assert metadata["weights_scope"] == "trainable"
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["checkpoint_monitor"] == "val/loss"
    assert history["epochs"][0]["checkpoint_selected"] is True
    assert history["epochs"][0]["global_step"] == 2
    assert "train/learning_rate" in history["epochs"][0]["metrics"]
    assert tracker.events[0][0]["selection/answerability_ap"] == result["val/answerability_ap"]
    assert tracker.events[0][0]["selection/epoch"] == 1.0


def test_warmup_ratio_scales_with_planned_training_steps() -> None:
    assert resolve_warmup_steps(101, {"warmup_ratio": 0.05, "warmup_steps": 200}) == 6
    assert resolve_warmup_steps(101, {"warmup_ratio": None, "warmup_steps": 20}) == 20
