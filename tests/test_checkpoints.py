import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_model
from torch import nn

from paper_vqa.training.checkpoints import CheckpointManager, CheckpointMetadata


class AdapterLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(2, 2)
        self.adapter = nn.Linear(2, 1)
        self.frozen.requires_grad_(False)


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(1, "val/answerability_ap", 0.9, None, {"model": "test"})


def test_new_checkpoint_contains_only_trainable_parameters(tmp_path: Path) -> None:
    model = AdapterLikeModel()
    expected_adapter = {
        name: value.detach().clone() for name, value in model.adapter.state_dict().items()
    }
    manager = CheckpointManager(tmp_path)

    manager.save(model, _metadata())

    tensor_names = set(load_file(str(manager.weights_path)))
    assert tensor_names == {"adapter.weight", "adapter.bias"}
    payload = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 2
    assert payload["weights_scope"] == "trainable"

    restored = AdapterLikeModel()
    restored.adapter.weight.data.zero_()
    restored.adapter.bias.data.zero_()
    metadata = manager.load(restored)

    assert metadata.weights_scope == "trainable"
    for name, value in restored.adapter.state_dict().items():
        assert torch.equal(value, expected_adapter[name])


def test_legacy_full_checkpoint_remains_loadable(tmp_path: Path) -> None:
    model = AdapterLikeModel()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    manager = CheckpointManager(tmp_path)
    save_model(model, str(manager.weights_path))
    legacy = asdict(_metadata())
    legacy.pop("format_version")
    legacy.pop("weights_scope")
    manager.metadata_path.write_text(json.dumps(legacy), encoding="utf-8")
    restored = AdapterLikeModel()
    for parameter in restored.parameters():
        parameter.data.zero_()

    metadata = manager.load(restored)

    assert metadata.weights_scope == "full"
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])
