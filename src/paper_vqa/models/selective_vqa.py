"""Multi-task BLIP VQA wrapper with calibrated abstention support."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from paper_vqa.models.answerability import AnswerabilityHead


@dataclass(frozen=True, slots=True)
class VQAForwardOutput:
    """Typed model output consumed by the multitask objective."""

    generation_logits: Tensor | None
    generation_loss: Tensor | None
    answerability_logits: Tensor | None
    answerability_probabilities: Tensor | None


@dataclass(frozen=True, slots=True)
class SelectivePrediction:
    """Inference result that distinguishes abstention from generated answers."""

    answer_ids: Tensor | None
    answerable_score: Tensor
    accepted: Tensor


@dataclass(frozen=True, slots=True)
class ScoredGeneration:
    """Generated answers with head and decoder confidence signals."""

    answer_ids: Tensor
    answerability_scores: Tensor | None
    decoder_scores: Tensor


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
        hidden_states: Tensor | None = None
        if self.answerability_head is not None:
            hidden_states = self._multimodal_hidden_states(pixel_values, input_ids, attention_mask)
            head_output = self.answerability_head(hidden_states, attention_mask)
            response_logits = head_output.logits
            response_probabilities = head_output.probabilities
        if labels is None:
            return VQAForwardOutput(
                generation_logits=None,
                generation_loss=None,
                answerability_logits=response_logits,
                answerability_probabilities=response_probabilities,
            )
        decoder_output = self._decoder_output(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            hidden_states=hidden_states,
        )
        return VQAForwardOutput(
            generation_logits=cast(Tensor, decoder_output.logits),
            generation_loss=cast(Tensor, decoder_output.loss).mean(),
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

    @torch.no_grad()
    def generate_with_scores(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        generation_kwargs: dict[str, Any],
    ) -> ScoredGeneration:
        """Generate every answer and expose comparable selection scores.

        This method is intended for scientific evaluation, where every policy
        must be compared on the same generated answer. Deployment inference
        should continue to use :meth:`predict`, which avoids decoder work for
        examples rejected by the answerability head.

        Args:
            pixel_values: Preprocessed image batch.
            input_ids: Tokenised questions.
            attention_mask: Question-token mask.
            generation_kwargs: BLIP decoding configuration.

        Returns:
            Generated token IDs, optional head probabilities, and geometric
            mean token probabilities from the decoder.
        """
        answerability_scores: Tensor | None = None
        if self.answerability_head is not None:
            hidden_states = self._multimodal_hidden_states(pixel_values, input_ids, attention_mask)
            answerability_scores = self.answerability_head(
                hidden_states, attention_mask
            ).probabilities[:, 1]
        options = dict(generation_kwargs)
        options["return_dict_in_generate"] = True
        options["output_scores"] = True
        generated = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **options,
        )
        sequences = cast(Tensor | None, getattr(generated, "sequences", None))
        generation_scores = getattr(generated, "scores", None)
        if sequences is None or not generation_scores:
            raise RuntimeError("BLIP generation did not return token-level scores")
        transition_scores = self.blip_model.text_decoder.compute_transition_scores(
            sequences,
            generation_scores,
            getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        generated_mask = transition_scores.ne(0)
        token_counts = generated_mask.sum(dim=1).clamp_min(1)
        mean_log_probability = transition_scores.sum(dim=1) / token_counts
        decoder_scores = mean_log_probability.exp().clamp(0.0, 1.0)
        return ScoredGeneration(sequences, answerability_scores, decoder_scores)

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

    def _decoder_output(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        hidden_states: Tensor | None,
    ) -> Any:
        """Run BLIP's answer decoder with unreduced loss and token logits.

        ``BlipForQuestionAnswering.forward`` intentionally returns only a mean
        decoder loss. The training objective needs logits to choose whether VQA
        loss applies to all examples or only answerable ones, so this mirrors
        BLIP's internal vision-to-text path and calls its decoder directly.

        Args:
            pixel_values: Preprocessed image tensor.
            input_ids: Tokenised question IDs.
            attention_mask: Question-token mask.
            labels: Padded answer IDs, with ignored positions set to ``-100``.
            hidden_states: Reusable multimodal states computed for the head.

        Returns:
            BLIP decoder output containing per-token logits and an unreduced loss.
        """
        multimodal_states = (
            hidden_states
            if hidden_states is not None
            else self._multimodal_hidden_states(pixel_values, input_ids, attention_mask)
        )
        decoder_ids = self._safe_decoder_input_ids(labels)
        return self.blip_model.text_decoder(
            input_ids=decoder_ids,
            encoder_hidden_states=multimodal_states,
            encoder_attention_mask=attention_mask,
            labels=labels,
            reduction="none",
            return_dict=True,
        )

    def _safe_decoder_input_ids(self, labels: Tensor | None) -> Tensor | None:
        """Replace ignored labels before decoder embedding lookup."""
        if labels is None:
            return None
        pad_token_id = int(self.blip_model.config.text_config.pad_token_id)
        return labels.masked_fill(labels == -100, pad_token_id)
