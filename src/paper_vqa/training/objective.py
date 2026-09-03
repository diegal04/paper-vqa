"""Configurable multitask VQA objective with explicit label masking."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from paper_vqa.models.selective_vqa import VQAForwardOutput

VQALossPolicy = Literal["all_examples", "answerable_only"]


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    """Loss components emitted for optimisation and experiment logging."""

    total: Tensor
    generation: Tensor
    answerability: Tensor
    answerability_count: int


class MultitaskObjective(nn.Module):
    """Compute ``L_vqa + lambda_answerability * L_answerability`` safely."""

    def __init__(
        self,
        answerability_weight: float,
        vqa_loss_policy: VQALossPolicy,
        answerability_class_weights: Tensor | None = None,
    ) -> None:
        """Initialise the configured loss policy.

        Args:
            answerability_weight: Auxiliary-loss multiplier.
            vqa_loss_policy: Whether unanswerable VizWiz examples train the decoder.
            answerability_class_weights: Optional class weights for the binary head.
        """
        super().__init__()
        if answerability_weight < 0:
            raise ValueError("answerability_weight must be non-negative")
        self.answerability_weight = answerability_weight
        self.vqa_loss_policy = vqa_loss_policy
        self.class_weights = answerability_class_weights
        self.register_buffer("_class_weights_buffer", answerability_class_weights)

    def forward(self, output: VQAForwardOutput, batch: dict[str, Tensor]) -> ObjectiveOutput:
        """Calculate both losses while respecting ``has_answerable``.

        Args:
            output: Typed model output.
            batch: Tokenised model batch containing labels and task flags.

        Returns:
            Differentiable total and individual loss values.
        """
        generation = self._generation_loss(output, batch["labels"], batch["answerable"])
        answerability, count = self._answerability_loss(
            output.answerability_logits,
            batch["answerable"],
            batch["has_answerable"],
            generation,
        )
        return ObjectiveOutput(
            total=generation + self.answerability_weight * answerability,
            generation=generation,
            answerability=answerability,
            answerability_count=count,
        )

    def _generation_loss(
        self, output: VQAForwardOutput, labels: Tensor, answerable: Tensor
    ) -> Tensor:
        """Compute per-example decoder loss to support the ablation policy."""
        logits = output.generation_logits
        if logits.shape[:2] != labels.shape:
            if self.vqa_loss_policy == "all_examples" and output.generation_loss is not None:
                return output.generation_loss
            raise ValueError("generation logits and labels must share [batch, sequence] dimensions")
        token_losses = functional.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )
        token_counts = (labels != -100).sum(dim=1).clamp_min(1)
        per_example = token_losses.sum(dim=1) / token_counts
        if self.vqa_loss_policy == "all_examples":
            return per_example.mean()
        mask = answerable.to(dtype=torch.bool)
        return per_example[mask].mean() if bool(mask.any()) else per_example.sum() * 0.0

    def _answerability_loss(
        self,
        logits: Tensor | None,
        labels: Tensor,
        has_answerable: Tensor,
        fallback: Tensor,
    ) -> tuple[Tensor, int]:
        """Apply the auxiliary loss only to examples carrying true labels."""
        mask = has_answerable.to(dtype=torch.bool)
        count = int(mask.sum().item())
        if logits is None or count == 0:
            return fallback.sum() * 0.0, count
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return functional.cross_entropy(logits[mask], labels[mask], weight=weights), count
