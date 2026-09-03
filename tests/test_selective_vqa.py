from types import SimpleNamespace

import torch
from torch import Tensor, nn

from paper_vqa.models.selective_vqa import SelectiveVQAModel


class _Vision(nn.Module):
    def forward(self, pixel_values: Tensor, return_dict: bool) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=pixel_values.unsqueeze(1))


class _TextEncoder(nn.Module):
    def forward(self, input_ids: Tensor, **_: object) -> SimpleNamespace:
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class _TextDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, input_ids: Tensor, labels: Tensor, **_: object) -> SimpleNamespace:
        self.calls += 1
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        return SimpleNamespace(loss=torch.ones(input_ids.shape[0]), logits=logits)


class _BlipStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = _Vision()
        self.text_encoder = _TextEncoder()
        self.text_decoder = _TextDecoder()
        self.config = SimpleNamespace(text_config=SimpleNamespace(pad_token_id=0))


def test_forward_uses_decoder_logits_only_when_generation_labels_exist() -> None:
    blip = _BlipStub()
    model = SelectiveVQAModel(blip, None)
    pixel_values = torch.ones((2, 3), dtype=torch.float32)
    input_ids = torch.tensor([[1, 2], [3, 4]])
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[5, -100], [6, 7]])

    generated = model(pixel_values, input_ids, attention_mask, labels)
    classified = model(pixel_values, input_ids, attention_mask, labels=None)

    assert generated.generation_logits is not None
    assert generated.generation_logits.shape == (2, 2, 8)
    assert generated.generation_loss is not None
    assert blip.text_decoder.calls == 1
    assert classified.generation_logits is None
    assert classified.generation_loss is None
