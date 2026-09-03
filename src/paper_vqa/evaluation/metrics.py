"""Metrics for answer quality, abstention quality and calibrated safety policies."""

import re
import string
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class AnswerabilityMetrics:
    """Classification and calibration metrics for answerability scores."""

    average_precision: float
    auroc: float
    f1: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    brier: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Validation-selected threshold and its achieved safety statistics."""

    threshold: float
    answerable_recall: float
    unsafe_answer_rate: float


@dataclass(frozen=True, slots=True)
class SelectiveMetrics:
    """Metrics for a system that may abstain instead of answering."""

    coverage: float
    accepted_vqa_accuracy: float
    selective_risk: float
    unsafe_answer_rate: float


def normalise_answer(answer: str) -> str:
    """Apply the standard VQA-style normalisation used before answer matching."""
    lowered = answer.lower().strip()
    without_articles = re.sub(r"\b(a|an|the)\b", " ", lowered)
    without_punctuation = without_articles.translate(str.maketrans("", "", string.punctuation))
    return " ".join(without_punctuation.split())


def official_vqa_accuracy(prediction: str, references: Sequence[str]) -> float:
    """Calculate the VQA leave-one-annotator-out accuracy for one prediction.

    Args:
        prediction: Generated answer.
        references: All human answers, normally ten for official VizWiz data.

    Returns:
        The official-style mean score over held-out annotators.
    """
    if not references:
        raise ValueError("references cannot be empty")
    prediction_normalised = normalise_answer(prediction)
    matches = sum(normalise_answer(reference) == prediction_normalised for reference in references)
    total = len(references)
    score_when_matching_held_out = min(1.0, max(matches - 1, 0) / 3.0)
    score_when_not_matching_held_out = min(1.0, matches / 3.0)
    return (
        matches * score_when_matching_held_out
        + (total - matches) * score_when_not_matching_held_out
    ) / total


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 15
) -> float:
    """Compute expected calibration error with equal-width confidence bins."""
    if bins < 1:
        raise ValueError("bins must be at least one")
    label_array = np.asarray(labels, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape != score_array.shape or label_array.size == 0:
        raise ValueError("labels and scores must be equally sized and non-empty")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        mask = (score_array >= lower) & (
            score_array <= upper if upper == 1.0 else score_array < upper
        )
        if not np.any(mask):
            continue
        confidence = float(score_array[mask].mean())
        accuracy = float(label_array[mask].mean())
        error += float(mask.mean()) * abs(confidence - accuracy)
    return error


def answerability_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float, ece_bins: int = 15
) -> AnswerabilityMetrics:
    """Calculate discriminative and calibration metrics for answerability.

    Positive labels and scores both mean ``answerable``.
    """
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if label_array.shape != score_array.shape or label_array.size == 0:
        raise ValueError("labels and scores must be equally sized and non-empty")
    if not np.isin(label_array, [0, 1]).all():
        raise ValueError("answerability labels must be binary")
    predictions = (score_array >= threshold).astype(np.int64)
    negatives = label_array == 0
    specificity = (
        float(((predictions == 0) & negatives).sum() / negatives.sum())
        if negatives.any()
        else float("nan")
    )
    auroc = (
        float(roc_auc_score(label_array, score_array))
        if np.unique(label_array).size == 2
        else float("nan")
    )
    return AnswerabilityMetrics(
        average_precision=float(average_precision_score(label_array, score_array)),
        auroc=auroc,
        f1=float(f1_score(label_array, predictions, zero_division=0)),
        precision=float(precision_score(label_array, predictions, zero_division=0)),
        recall=float(recall_score(label_array, predictions, zero_division=0)),
        specificity=specificity,
        balanced_accuracy=float(balanced_accuracy_score(label_array, predictions)),
        brier=float(np.mean((score_array - label_array) ** 2)),
        expected_calibration_error=expected_calibration_error(labels, scores, ece_bins),
    )


def select_safety_threshold(
    labels: Sequence[int], scores: Sequence[float], minimum_answerable_recall: float
) -> ThresholdResult:
    """Choose the safest validation threshold satisfying a recall constraint.

    Args:
        labels: Binary answerability labels where one is answerable.
        scores: Predicted answerable probabilities.
        minimum_answerable_recall: Required recall on genuinely answerable questions.

    Returns:
        The threshold that minimises responses to unanswerable questions.
    """
    if not 0.0 <= minimum_answerable_recall <= 1.0:
        raise ValueError("minimum_answerable_recall must be in [0, 1]")
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if not np.any(label_array == 1):
        raise ValueError("threshold calibration needs at least one answerable example")
    candidates = np.unique(np.concatenate(([0.0, 1.0], score_array)))
    viable: list[ThresholdResult] = []
    for threshold in candidates:
        accepted = score_array >= threshold
        recall = float(accepted[label_array == 1].mean())
        unsafe = float(accepted[label_array == 0].mean()) if np.any(label_array == 0) else 0.0
        if recall >= minimum_answerable_recall:
            viable.append(ThresholdResult(float(threshold), recall, unsafe))
    if not viable:
        raise RuntimeError("no threshold satisfies the configured answerable recall")
    return min(viable, key=lambda result: (result.unsafe_answer_rate, -result.threshold))


def selective_metrics(
    accepted: Sequence[bool], vqa_scores: Sequence[float], answerability_labels: Sequence[int]
) -> SelectiveMetrics:
    """Measure coverage, accepted-answer accuracy and unsafe response frequency."""
    accept_array = np.asarray(accepted, dtype=bool)
    vqa_array = np.asarray(vqa_scores, dtype=np.float64)
    labels = np.asarray(answerability_labels, dtype=np.int64)
    if not (accept_array.shape == vqa_array.shape == labels.shape) or accept_array.size == 0:
        raise ValueError("selective metric inputs must be equally sized and non-empty")
    coverage = float(accept_array.mean())
    accepted_accuracy = (
        float(vqa_array[accept_array].mean()) if accept_array.any() else float("nan")
    )
    risk = 1.0 - accepted_accuracy if accept_array.any() else float("nan")
    unanswerable = labels == 0
    unsafe = float(accept_array[unanswerable].mean()) if unanswerable.any() else 0.0
    return SelectiveMetrics(coverage, accepted_accuracy, risk, unsafe)


def aggregate_seed_metrics(metrics: Sequence[AnswerabilityMetrics]) -> dict[str, dict[str, float]]:
    """Aggregate dataclass metrics over seeds into mean and sample deviation."""
    if not metrics:
        raise ValueError("at least one seed result is required")
    values = [asdict(metric) for metric in metrics]
    aggregate: dict[str, dict[str, float]] = {}
    for name in values[0]:
        per_seed = [item[name] for item in values]
        aggregate[name] = {
            "mean": float(np.nanmean(per_seed)),
            "std": float(np.nanstd(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        }
    return aggregate


def bootstrap_confidence_interval(
    values: Sequence[float], seed: int, samples: int = 1_000, confidence: float = 0.95
) -> tuple[float, float]:
    """Bootstrap a percentile confidence interval for a mean metric."""
    if not values or samples < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap arguments")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = generator.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))
