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
        unanswerable_vqa_weight: float = 1.0,
    ) -> None:
        """Initialise the configured loss policy.

        Args:
            answerability_weight: Auxiliary-loss multiplier.
            vqa_loss_policy: Whether unanswerable VizWiz examples train the decoder.
            answerability_class_weights: Optional class weights for the binary head.
            unanswerable_vqa_weight: Decoder-loss multiplier for explicitly
                unanswerable records under ``all_examples``.
        """
        super().__init__()
        if answerability_weight < 0:
            raise ValueError("answerability_weight must be non-negative")
        if unanswerable_vqa_weight < 0:
            raise ValueError("unanswerable_vqa_weight must be non-negative")
        self.answerability_weight = answerability_weight
        self.vqa_loss_policy = vqa_loss_policy
        self.unanswerable_vqa_weight = unanswerable_vqa_weight
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
        generation = self._generation_loss(
            output,
            batch["labels"],
            batch["answerable"],
            batch["has_answerable"],
        )
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
        self,
        output: VQAForwardOutput,
        labels: Tensor,
        answerable: Tensor,
        has_answerable: Tensor,
    ) -> Tensor:
        """Compute next-token decoder loss under the configured sample policy.

        BLIP's decoder is autoregressive: logits at position ``t`` predict the
        label at ``t + 1``. This explicit shift reproduces its native language
        modelling objective while allowing unanswerable records to be excluded
        for the ``answerable_only`` ablation.
        """
        logits = output.generation_logits
        if logits is None or logits.shape[:2] != labels.shape:
            if self.vqa_loss_policy == "all_examples" and output.generation_loss is not None:
                return output.generation_loss
            raise ValueError("generation logits and labels must share [batch, sequence] dimensions")
        if labels.shape[1] < 2:
            return logits.sum() * 0.0
        token_losses = functional.cross_entropy(
            logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100, reduction="none"
        )
        valid_tokens = labels[:, 1:] != -100
        explicitly_unanswerable = has_answerable.bool() & ~answerable.bool()
        negative_weight = (
            0.0 if self.vqa_loss_policy == "answerable_only" else self.unanswerable_vqa_weight
        )
        sample_weights = torch.ones(
            labels.shape[0], dtype=token_losses.dtype, device=token_losses.device
        )
        sample_weights[explicitly_unanswerable] = negative_weight
        token_weights = valid_tokens.to(token_losses.dtype) * sample_weights.unsqueeze(1)
        denominator = token_weights.sum()
        return (
            (token_losses * token_weights).sum() / denominator.clamp_min(1.0)
            if bool(denominator > 0)
            else token_losses.sum() * 0.0
        )

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
