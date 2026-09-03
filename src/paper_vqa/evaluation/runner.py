"""End-to-end frozen-test evaluation after validation-only threshold calibration."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from paper_vqa.data.datasets import load_rgb_image
from paper_vqa.data.records import VQAExample
from paper_vqa.evaluation.metrics import (
    AnswerabilityMetrics,
    SelectiveMetrics,
    answerability_metrics,
    official_vqa_accuracy,
    selective_metrics,
)
from paper_vqa.evaluation.reporting import PredictionRecord, write_predictions
from paper_vqa.models.selective_vqa import SelectiveVQAModel


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Frozen-test result bundle written as a paper-ready JSON artefact."""

    threshold: float
    vqa_accuracy: float
    answerability: AnswerabilityMetrics | None
    selective: SelectiveMetrics | None


class Evaluator:
    """Calibrate on validation and evaluate a checkpoint once on a supplied test set."""

    def __init__(self, model: SelectiveVQAModel, processor: Any, device: str) -> None:
        """Store a loaded model and its matching processor."""
        self.model = model.to(device).eval()
        self.processor = processor
        self.device = torch.device(device)

    @torch.no_grad()
    def calibration_scores(
        self, loader: DataLoader[dict[str, Tensor]]
    ) -> tuple[list[int], list[float]]:
        """Collect labelled validation scores without selecting a decision threshold."""
        if self.model.answerability_head is None:
            return [], []
        labels: list[int] = []
        scores: list[float] = []
        for batch in loader:
            moved = {name: value.to(self.device) for name, value in batch.items()}
            output = self.model(
                pixel_values=moved["pixel_values"],
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                labels=None,
            )
            assert output.answerability_probabilities is not None
            mask = moved["has_answerable"].bool()
            labels.extend(moved["answerable"][mask].cpu().tolist())
            scores.extend(output.answerability_probabilities[mask, 1].cpu().tolist())
        return labels, scores

    @torch.no_grad()
    def evaluate_examples(
        self,
        examples: Sequence[VQAExample],
        threshold: float,
        generation_kwargs: dict[str, Any],
        ece_bins: int,
    ) -> tuple[EvaluationResult, list[PredictionRecord]]:
        """Run selective inference over frozen test examples and score all outputs."""
        records: list[PredictionRecord] = []
        vqa_scores: list[float] = []
        accepted: list[bool] = []
        labels: list[int] = []
        probabilities: list[float] = []
        for example in examples:
            inputs = self.processor(
                images=load_rgb_image(example.image),
                text=example.question,
                return_tensors="pt",
            ).to(self.device)
            prediction = self.model.predict(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                threshold=threshold,
                generation_kwargs=generation_kwargs,
            )
            is_accepted = bool(prediction.accepted.item())
            answer = (
                self.processor.decode(prediction.answer_ids[0], skip_special_tokens=True)
                if is_accepted and prediction.answer_ids is not None
                else None
            )
            score = float(prediction.answerable_score.item())
            records.append(
                PredictionRecord(
                    sample_id=example.sample_id,
                    prediction=answer,
                    answerable_score=score,
                    accepted=is_accepted,
                    references=example.answers,
                    answerable=example.answerable,
                )
            )
            accepted.append(is_accepted)
            score_vqa = (
                official_vqa_accuracy(answer, example.answers) if answer is not None else 0.0
            )
            vqa_scores.append(score_vqa)
            if example.answerable is not None:
                labels.append(example.answerable)
                probabilities.append(score)
        vqa_accuracy = float(sum(vqa_scores) / len(vqa_scores)) if vqa_scores else float("nan")
        classification = (
            answerability_metrics(labels, probabilities, threshold, ece_bins)
            if labels and self.model.answerability_head is not None
            else None
        )
        selective = selective_metrics(accepted, vqa_scores, labels) if labels else None
        return EvaluationResult(threshold, vqa_accuracy, classification, selective), records


def write_evaluation(
    result: EvaluationResult,
    records: Sequence[PredictionRecord],
    directory: Path,
) -> None:
    """Write metrics and full per-example predictions for a final evaluation."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_predictions(records, directory / "predictions.json")
