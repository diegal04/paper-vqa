from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from paper_vqa.models.selective_vqa import VQAForwardOutput
from paper_vqa.training.objective import MultitaskObjective
from paper_vqa.training.tracker import NullTracker
from paper_vqa.training.trainer import Trainer, build_optimizer


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


def test_trainer_smoke_saves_safe_checkpoint(tmp_path: Path) -> None:
    model = TinyVQA()
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
        NullTracker(),
        config,
        {"smoke": True},
    )
    loader = DataLoader(TinyDataset(), batch_size=2)

    result = trainer.fit(loader, loader)

    assert "val/loss" in result
    assert (tmp_path / "checkpoint" / "model.safetensors").exists()
    assert (tmp_path / "checkpoint" / "metadata.json").exists()
