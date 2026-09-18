"""End-to-end frozen-test evaluation after validation-only threshold calibration."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from paper_vqa.data.datasets import load_rgb_image
from paper_vqa.data.records import VQAExample
from paper_vqa.evaluation.metrics import (
    AnswerabilityMetrics,
    SelectiveMetrics,
    answerability_metrics,
    official_vqa_accuracy,
    selective_metrics,
)
from paper_vqa.evaluation.reporting import GenerationRecord, PredictionRecord, write_predictions
from paper_vqa.models.selective_vqa import SelectiveVQAModel


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Frozen-test result bundle written as a paper-ready JSON artefact."""

    threshold: float
    vqa_accuracy: float
    answerability: AnswerabilityMetrics | None
    selective: SelectiveMetrics | None


def flatten_evaluation_metrics(result: EvaluationResult, prefix: str) -> dict[str, float]:
    """Flatten every available evaluation scalar for experiment tracking.

    Args:
        result: Structured VQA, answerability, and selective-evaluation output.
        prefix: Split namespace such as ``validation`` or ``test``.

    Returns:
        Flat slash-delimited metric names suitable for W&B logging.
    """
    flattened: dict[str, float] = {}

    def visit(value: Any, path: str) -> None:
        """Recursively collect numeric leaves while omitting unavailable groups."""
        if value is None:
            return
        if isinstance(value, Mapping):
            for name, child in value.items():
                visit(child, f"{path}/{name}")
            return
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                flattened[path] = numeric

    visit(asdict(result), prefix.rstrip("/"))
    return flattened


class Evaluator:
    """Calibrate on validation and evaluate a checkpoint once on a supplied test set."""

    def __init__(
        self,
        model: SelectiveVQAModel,
        processor: Any,
        device: str,
        progress_enabled: bool = True,
        progress_leave: bool = False,
    ) -> None:
        """Store a loaded model, processor, and terminal-progress settings.

        Args:
            model: Model to evaluate.
            processor: Matching BLIP processor.
            device: Device identifier for inference.
            progress_enabled: Whether terminal progress bars are displayed.
            progress_leave: Whether completed bars remain in the terminal.
        """
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = torch.device(device)
        self.progress_enabled = progress_enabled
        self.progress_leave = progress_leave

    @torch.no_grad()
    def generate_examples(
        self,
        examples: Sequence[VQAExample],
        generation_kwargs: dict[str, Any],
    ) -> list[GenerationRecord]:
        """Generate every answer once with head and decoder confidence scores.

        Args:
            examples: Labelled or unlabelled VQA records.
            generation_kwargs: BLIP generation settings.

        Returns:
            Policy-independent generations reusable by every abstention baseline.
        """
        records: list[GenerationRecord] = []
        for example in tqdm(
            examples,
            desc="Generating scored predictions",
            total=len(examples),
            disable=not self.progress_enabled,
            leave=self.progress_leave,
        ):
            inputs = self.processor(
                images=load_rgb_image(example.image),
                text=example.question,
                return_tensors="pt",
            ).to(self.device)
            generated = self.model.generate_with_scores(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_kwargs=generation_kwargs,
            )
            answer = self.processor.decode(generated.answer_ids[0], skip_special_tokens=True)
            answerability_score = (
                float(generated.answerability_scores[0].item())
                if generated.answerability_scores is not None
                else None
            )
            records.append(
                GenerationRecord(
                    sample_id=example.sample_id,
                    generated_answer=answer,
                    answerability_score=answerability_score,
                    decoder_score=float(generated.decoder_scores[0].item()),
                    references=example.answers,
                    answerable=example.answerable,
                )
            )
        return records

    def evaluate_records(
        self,
        generations: Sequence[GenerationRecord],
        threshold: float,
        ece_bins: int,
    ) -> tuple[EvaluationResult, list[PredictionRecord]]:
        """Apply the head-only operating point to policy-independent generations.

        Args:
            generations: Outputs produced once for all comparison policies.
            threshold: Validation-selected answerability-head threshold.
            ece_bins: Number of equal-width calibration bins.

        Returns:
            Main head-only result and auditable per-example predictions.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        records: list[PredictionRecord] = []
        vqa_scores: list[float] = []
        labels: list[int] = []
        probabilities: list[float] = []
        labelled_accepted: list[bool] = []
        labelled_vqa_scores: list[float] = []
        labelled_answers: list[str | None] = []
        for generation in generations:
            is_accepted = (
                generation.answerability_score is None
                or generation.answerability_score >= threshold
            )
            answer = generation.generated_answer if is_accepted else None
            records.append(
                PredictionRecord(
                    sample_id=generation.sample_id,
                    prediction=answer,
                    generated_answer=generation.generated_answer,
                    answerability_score=generation.answerability_score,
                    decoder_score=generation.decoder_score,
                    accepted=is_accepted,
                    references=generation.references,
                    answerable=generation.answerable,
                )
            )
            score_vqa = (
                official_vqa_accuracy(answer, generation.references) if answer is not None else 0.0
            )
            vqa_scores.append(score_vqa)
            if generation.answerable is not None:
                labels.append(generation.answerable)
                labelled_accepted.append(is_accepted)
                labelled_vqa_scores.append(score_vqa)
                labelled_answers.append(answer)
                if generation.answerability_score is not None:
                    probabilities.append(generation.answerability_score)
        vqa_accuracy = float(sum(vqa_scores) / len(vqa_scores)) if vqa_scores else float("nan")
        classification = (
            answerability_metrics(labels, probabilities, threshold, ece_bins)
            if labels and len(probabilities) == len(labels)
            else None
        )
        selective = (
            selective_metrics(labelled_accepted, labelled_vqa_scores, labels, labelled_answers)
            if labels
            else None
        )
        return EvaluationResult(threshold, vqa_accuracy, classification, selective), records


def write_evaluation(
    result: EvaluationResult,
    records: Sequence[PredictionRecord],
    directory: Path,
) -> None:
    """Write metrics and full per-example predictions for a final evaluation."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(asdict(result))
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_predictions(records, directory / "predictions.json")


def _json_safe(value: Any) -> Any:
    """Replace non-finite metric leaves with JSON ``null`` recursively."""
    if isinstance(value, dict):
        return {name: _json_safe(child) for name, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
