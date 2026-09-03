"""Answerability classification head over BLIP multimodal representations."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

PoolingStrategy = Literal["masked_mean", "cls"]


@dataclass(frozen=True, slots=True)
class AnswerabilityOutput:
    """Logits and probabilities predicted by the answerability head."""

    logits: Tensor
    probabilities: Tensor


class AnswerabilityHead(nn.Module):
    """Binary MLP that estimates whether an image-question pair is answerable."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        pooling: PoolingStrategy = "masked_mean",
    ) -> None:
        """Initialise a configurable head and its explicit pooling strategy.

        Args:
            input_dim: Multimodal encoder hidden size.
            hidden_dim: Width of the first classifier layer.
            dropout: Dropout probability in the classifier.
            pooling: ``masked_mean`` excludes question padding; ``cls`` uses token zero.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Use a documented Xavier initialisation for new linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> AnswerabilityOutput:
        """Classify answerability from multimodal text-encoder states.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]``.
            attention_mask: Binary question mask of shape ``[batch, sequence]``.

        Returns:
            Answerability logits ordered as ``[unanswerable, answerable]``.
        """
        pooled = self.pool(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        return AnswerabilityOutput(logits=logits, probabilities=torch.softmax(logits, dim=-1))

    def pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool sequence states without allowing padding to alter mean features."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the first two hidden-state dimensions")
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denominator
