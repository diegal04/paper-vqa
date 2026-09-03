"""Multi-task BLIP VQA wrapper with calibrated abstention support."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from paper_vqa.models.answerability import AnswerabilityHead


@dataclass(frozen=True, slots=True)
class VQAForwardOutput:
    """Typed model output consumed by the multitask objective."""

    generation_logits: Tensor
    generation_loss: Tensor | None
    answerability_logits: Tensor | None
    answerability_probabilities: Tensor | None


@dataclass(frozen=True, slots=True)
class SelectivePrediction:
    """Inference result that distinguishes abstention from generated answers."""

    answer_ids: Tensor | None
    answerable_score: Tensor
    accepted: Tensor


class SelectiveVQAModel(nn.Module):
    """BLIP VQA with optional LoRA and an optional answerability head."""

    def __init__(self, blip_model: nn.Module, answerability_head: AnswerabilityHead | None) -> None:
        """Wrap a BLIP question-answering model.

        Args:
            blip_model: ``BlipForQuestionAnswering`` optionally wrapped by PEFT.
            answerability_head: New auxiliary classifier, or ``None`` for VQA-only ablation.
        """
        super().__init__()
        self.blip_model: Any = blip_model
        self.answerability_head = answerability_head

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
    ) -> VQAForwardOutput:
        """Run generation and, when enabled, answerability classification."""
        response_logits: Tensor | None = None
        response_probabilities: Tensor | None = None
        if self.answerability_head is not None:
            hidden_states = self._multimodal_hidden_states(pixel_values, input_ids, attention_mask)
            head_output = self.answerability_head(hidden_states, attention_mask)
            response_logits = head_output.logits
            response_probabilities = head_output.probabilities
        decoder_ids = self._safe_decoder_input_ids(labels)
        blip_output = self.blip_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_ids,
            labels=labels,
            return_dict=True,
        )
        return VQAForwardOutput(
            generation_logits=blip_output.logits,
            generation_loss=blip_output.loss,
            answerability_logits=response_logits,
            answerability_probabilities=response_probabilities,
        )

    def predict(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        threshold: float,
        generation_kwargs: dict[str, Any],
    ) -> SelectivePrediction:
        """Classify first and generate only for examples accepted by the threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.answerability_head is None:
            score = torch.ones(pixel_values.shape[0], device=pixel_values.device)
        else:
            hidden_states = self._multimodal_hidden_states(pixel_values, input_ids, attention_mask)
            score = self.answerability_head(hidden_states, attention_mask).probabilities[:, 1]
        accepted = score >= threshold
        if not bool(accepted.any()):
            return SelectivePrediction(None, score, accepted)
        answer_ids = self.blip_model.generate(
            pixel_values=pixel_values[accepted],
            input_ids=input_ids[accepted],
            attention_mask=attention_mask[accepted],
            **generation_kwargs,
        )
        return SelectivePrediction(answer_ids, score, accepted)

    def _multimodal_hidden_states(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Compute BLIP text features cross-attending to visual encoder features."""
        vision_outputs = self.blip_model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_mask = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device
        )
        text_outputs = self.blip_model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_mask,
            return_dict=True,
        )
        return cast(Tensor, text_outputs.last_hidden_state)

    def _safe_decoder_input_ids(self, labels: Tensor | None) -> Tensor | None:
        """Replace ignored labels before decoder embedding lookup."""
        if labels is None:
            return None
        pad_token_id = int(self.blip_model.config.text_config.pad_token_id)
        return labels.masked_fill(labels == -100, pad_token_id)
