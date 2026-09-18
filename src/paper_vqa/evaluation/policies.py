"""First-class abstention policies and risk--coverage reporting."""

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from paper_vqa.evaluation.metrics import (
    SelectiveMetrics,
    normalise_answer,
    official_vqa_accuracy,
    select_safety_threshold,
    selective_metrics,
)
from paper_vqa.evaluation.reporting import GenerationRecord


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Metrics at one validation-calibrated policy operating point."""

    threshold: float | None
    vqa_accuracy: float
    selective: SelectiveMetrics


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """One confidence threshold on a selective risk--coverage curve."""

    threshold: float
    coverage: float
    accepted_vqa_accuracy: float | None
    selective_risk: float | None
    answerable_recall: float
    unsafe_answer_rate: float
    specific_unsafe_answer_rate: float


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    """Operating-point metrics and curves for all available policies."""

    policies: dict[str, PolicyEvaluation]
    curves: dict[str, tuple[RiskCoveragePoint, ...]]
    matched_answerable_recall: dict[str, tuple["MatchedRecallPoint", ...]]


@dataclass(frozen=True, slots=True)
class MatchedRecallThresholds:
    """Validation-calibrated thresholds for one emitted-answer recall target."""

    target_answerable_recall: float
    decoder_confidence: float | None
    head: float | None


@dataclass(frozen=True, slots=True)
class MatchedRecallPoint:
    """Policy performance at a validation-frozen answerable-emission recall target."""

    target_answerable_recall: float
    threshold: float
    achieved_answerable_recall: float
    coverage: float
    specific_unsafe_answer_rate: float
    emission_answerability_precision: float
    accepted_vqa_accuracy: float
    accepted_count: int
    specific_unsafe_count: int


@dataclass(frozen=True, slots=True)
class CalibratedThresholds:
    """Validation-only thresholds used on development or frozen test."""

    head: float | None
    decoder_confidence: float
    matched_answerable_recall: tuple[MatchedRecallThresholds, ...] = ()


def calibrate_policy_thresholds(
    records: Sequence[GenerationRecord],
    minimum_answerable_recall: float,
    matched_answerable_recall_targets: Sequence[float] = (),
) -> CalibratedThresholds:
    """Calibrate head and decoder-confidence thresholds from validation records.

    Args:
        records: Unconditionally generated validation predictions.
        minimum_answerable_recall: Required recall before textual decoder abstention.
        matched_answerable_recall_targets: Emitted-answer recall targets used to
            compare decoder confidence and the auxiliary head after the literal gate.

    Returns:
        Independent thresholds for the available confidence signals.
    """
    labelled = [record for record in records if record.answerable is not None]
    if not labelled:
        raise ValueError("policy calibration requires answerability labels")
    labels = [_answerability_label(record) for record in labelled]
    decoder = select_safety_threshold(
        labels,
        [record.decoder_score for record in labelled],
        minimum_answerable_recall,
    ).threshold
    head_scores = [record.answerability_score for record in labelled]
    head = None
    if all(score is not None for score in head_scores):
        head = select_safety_threshold(
            labels,
            [float(score) for score in head_scores if score is not None],
            minimum_answerable_recall,
        ).threshold
    matched = tuple(
        MatchedRecallThresholds(
            target_answerable_recall=target,
            decoder_confidence=_select_emission_threshold(
                labelled,
                [record.decoder_score for record in labelled],
                target,
            ),
            head=(
                _select_emission_threshold(
                    labelled,
                    [float(score) for score in head_scores if score is not None],
                    target,
                )
                if head is not None
                else None
            ),
        )
        for target in _validated_recall_targets(matched_answerable_recall_targets)
    )
    return CalibratedThresholds(
        head=head,
        decoder_confidence=decoder,
        matched_answerable_recall=matched,
    )


def compare_abstention_policies(
    records: Sequence[GenerationRecord],
    thresholds: CalibratedThresholds,
    maximum_curve_points: int = 101,
) -> PolicyComparison:
    """Evaluate always-answer, decoder, head, and hybrid selection fairly.

    All policies reuse exactly the same generated answers. Only the decision to
    emit or abstain changes, which isolates selection from decoder stochasticity.

    Args:
        records: Generated predictions with confidence scores and references.
        thresholds: Thresholds calibrated exclusively on validation.
        maximum_curve_points: Maximum thresholds retained for each plotted curve.

    Returns:
        Policy operating points and sampled risk--coverage curves.
    """
    if maximum_curve_points < 2:
        raise ValueError("maximum_curve_points must be at least two")
    labelled = tuple(record for record in records if record.answerable is not None)
    if not labelled:
        raise ValueError("policy comparison requires answerability labels")
    literal_answer = tuple(
        normalise_answer(record.generated_answer) == "unanswerable" for record in labelled
    )
    decisions: dict[str, tuple[bool, ...]] = {
        "always_answer": tuple(True for _ in labelled),
        "decoder_literal": tuple(not literal for literal in literal_answer),
        "decoder_confidence": tuple(
            record.decoder_score >= thresholds.decoder_confidence for record in labelled
        ),
        "decoder_confidence_plus_literal": tuple(
            record.decoder_score >= thresholds.decoder_confidence and not literal
            for record, literal in zip(labelled, literal_answer, strict=True)
        ),
    }
    if thresholds.head is not None:
        head_decisions = tuple(
            record.answerability_score is not None and record.answerability_score >= thresholds.head
            for record in labelled
        )
        decisions["head_only"] = head_decisions
        decisions["hybrid"] = tuple(
            accepted and not literal
            for accepted, literal in zip(head_decisions, literal_answer, strict=True)
        )
    threshold_by_policy = {
        "always_answer": None,
        "decoder_literal": None,
        "decoder_confidence": thresholds.decoder_confidence,
        "decoder_confidence_plus_literal": thresholds.decoder_confidence,
        "head_only": thresholds.head,
        "hybrid": thresholds.head,
    }
    policies = {
        name: _evaluate_decisions(labelled, accepted, threshold_by_policy[name])
        for name, accepted in decisions.items()
    }
    curves = {
        "decoder_confidence": _risk_coverage_curve(
            labelled,
            tuple(record.decoder_score for record in labelled),
            lambda _record: True,
            maximum_curve_points,
        ),
        "decoder_confidence_plus_literal": _risk_coverage_curve(
            labelled,
            tuple(record.decoder_score for record in labelled),
            lambda record: normalise_answer(record.generated_answer) != "unanswerable",
            maximum_curve_points,
        ),
    }
    if thresholds.head is not None:
        head_scores = tuple(_head_score(record) for record in labelled)
        curves["head_only"] = _risk_coverage_curve(
            labelled, head_scores, lambda _record: True, maximum_curve_points
        )
        curves["hybrid"] = _risk_coverage_curve(
            labelled,
            head_scores,
            lambda record: normalise_answer(record.generated_answer) != "unanswerable",
            maximum_curve_points,
        )
    matched = _evaluate_matched_recall(labelled, thresholds.matched_answerable_recall)
    return PolicyComparison(
        policies=policies,
        curves=curves,
        matched_answerable_recall=matched,
    )


def flatten_policy_metrics(comparison: PolicyComparison, prefix: str) -> dict[str, float]:
    """Flatten policy operating points for W&B scalar logging."""
    metrics: dict[str, float] = {}
    for policy_name, evaluation in comparison.policies.items():
        base = f"{prefix.rstrip('/')}/policies/{policy_name}"
        if evaluation.threshold is not None:
            metrics[f"{base}/threshold"] = evaluation.threshold
        metrics[f"{base}/vqa_accuracy"] = evaluation.vqa_accuracy
        for name, value in asdict(evaluation.selective).items():
            if math.isfinite(value):
                metrics[f"{base}/{name}"] = value
    for policy_name, points in comparison.matched_answerable_recall.items():
        for point in points:
            target = f"target_{point.target_answerable_recall:.3f}"
            base = f"{prefix.rstrip('/')}/matched_answerable_recall/{target}/{policy_name}"
            for name, value in asdict(point).items():
                if name != "target_answerable_recall" and math.isfinite(value):
                    metrics[f"{base}/{name}"] = float(value)
    return metrics


def write_policy_comparison(comparison: PolicyComparison, directory: Path) -> None:
    """Write JSON, CSV, and dependency-free vector risk--coverage outputs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy_metrics.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "policies": {
                        name: asdict(value) for name, value in comparison.policies.items()
                    },
                    "matched_answerable_recall": {
                        name: [asdict(point) for point in points]
                        for name, points in comparison.matched_answerable_recall.items()
                    },
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_curve_csv(comparison, directory / "risk_coverage.csv")
    _write_matched_recall_csv(comparison, directory / "matched_answerable_recall.csv")
    _write_curve_svg(comparison, directory / "risk_coverage.svg")


def _evaluate_decisions(
    records: Sequence[GenerationRecord],
    accepted: Sequence[bool],
    threshold: float | None,
) -> PolicyEvaluation:
    """Compute selective metrics for one fixed decision vector."""
    answers = [
        record.generated_answer if emit else None
        for record, emit in zip(records, accepted, strict=True)
    ]
    vqa_scores = [
        official_vqa_accuracy(answer, record.references) if answer is not None else 0.0
        for record, answer in zip(records, answers, strict=True)
    ]
    labels = [_answerability_label(record) for record in records]
    selective = selective_metrics(accepted, vqa_scores, labels, answers)
    return PolicyEvaluation(
        threshold=threshold,
        vqa_accuracy=float(np.mean(vqa_scores)),
        selective=selective,
    )


def _validated_recall_targets(targets: Sequence[float]) -> tuple[float, ...]:
    """Validate and deterministically order matched answerable-recall targets."""
    values = tuple(float(target) for target in targets)
    if any(not 0.0 < target <= 1.0 for target in values):
        raise ValueError("matched answerable-recall targets must be in (0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("matched answerable-recall targets must be unique")
    return tuple(sorted(values))


def _select_emission_threshold(
    records: Sequence[GenerationRecord],
    scores: Sequence[float],
    target_answerable_recall: float,
) -> float | None:
    """Calibrate a confidence threshold after literal abstentions are removed.

    Args:
        records: Labelled validation generations.
        scores: Confidence signal associated with each generation.
        target_answerable_recall: Minimum fraction of genuinely answerable
            questions that must emit a non-literal answer.

    Returns:
        Safest viable threshold, or ``None`` when the decoder's literal
        abstentions make the requested recall unattainable.
    """
    labels = np.asarray([_answerability_label(record) for record in records], dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if labels.shape != score_array.shape or labels.size == 0:
        raise ValueError("matched-recall records and scores must be equally sized and non-empty")
    answerable = labels == 1
    if not answerable.any():
        raise ValueError("matched-recall calibration needs answerable examples")
    eligible = np.asarray(
        [normalise_answer(record.generated_answer) != "unanswerable" for record in records],
        dtype=bool,
    )
    values = np.unique(score_array)
    candidates = np.concatenate(([np.nextafter(values[-1], math.inf)], values))
    viable: list[tuple[float, float]] = []
    unanswerable = labels == 0
    for threshold in candidates:
        accepted = eligible & (score_array >= threshold)
        recall = float(accepted[answerable].mean())
        if recall < target_answerable_recall:
            continue
        unsafe = float(accepted[unanswerable].mean()) if unanswerable.any() else 0.0
        viable.append((unsafe, float(threshold)))
    if not viable:
        return None
    return min(viable, key=lambda result: (result[0], -result[1]))[1]


def _evaluate_matched_recall(
    records: Sequence[GenerationRecord],
    thresholds: Sequence[MatchedRecallThresholds],
) -> dict[str, tuple[MatchedRecallPoint, ...]]:
    """Apply validation-frozen matched-recall thresholds to evaluation records."""
    points: dict[str, list[MatchedRecallPoint]] = {
        "decoder_confidence_plus_literal": [],
        "hybrid": [],
    }
    for matched in thresholds:
        if matched.decoder_confidence is not None:
            points["decoder_confidence_plus_literal"].append(
                _matched_recall_point(
                    records,
                    tuple(record.decoder_score for record in records),
                    matched.decoder_confidence,
                    matched.target_answerable_recall,
                )
            )
        if matched.head is not None:
            points["hybrid"].append(
                _matched_recall_point(
                    records,
                    tuple(_head_score(record) for record in records),
                    matched.head,
                    matched.target_answerable_recall,
                )
            )
    return {name: tuple(values) for name, values in points.items() if values}


def _matched_recall_point(
    records: Sequence[GenerationRecord],
    scores: Sequence[float],
    threshold: float,
    target_answerable_recall: float,
) -> MatchedRecallPoint:
    """Compute one policy row using a threshold frozen on validation."""
    labels = np.asarray([_answerability_label(record) for record in records], dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    literal = np.asarray(
        [normalise_answer(record.generated_answer) == "unanswerable" for record in records],
        dtype=bool,
    )
    accepted = (score_array >= threshold) & ~literal
    answerable = labels == 1
    unanswerable = labels == 0
    accepted_count = int(accepted.sum())
    specific_unsafe_count = int((accepted & unanswerable).sum())
    vqa_scores = np.asarray(
        [official_vqa_accuracy(record.generated_answer, record.references) for record in records],
        dtype=np.float64,
    )
    accepted_vqa = float(vqa_scores[accepted].mean()) if accepted_count else float("nan")
    accepted_answerable = int((accepted & answerable).sum())
    return MatchedRecallPoint(
        target_answerable_recall=target_answerable_recall,
        threshold=threshold,
        achieved_answerable_recall=(
            float(accepted[answerable].mean()) if answerable.any() else float("nan")
        ),
        coverage=float(accepted.mean()),
        specific_unsafe_answer_rate=(
            float(specific_unsafe_count / unanswerable.sum())
            if unanswerable.any()
            else float("nan")
        ),
        emission_answerability_precision=(
            float(accepted_answerable / accepted_count) if accepted_count else float("nan")
        ),
        accepted_vqa_accuracy=accepted_vqa,
        accepted_count=accepted_count,
        specific_unsafe_count=specific_unsafe_count,
    )


def _risk_coverage_curve(
    records: Sequence[GenerationRecord],
    scores: Sequence[float],
    eligible: Callable[[GenerationRecord], bool],
    maximum_points: int,
) -> tuple[RiskCoveragePoint, ...]:
    """Sweep a confidence score while preserving an optional textual gate."""
    thresholds = _sample_thresholds(scores, maximum_points)
    labels = np.asarray([_answerability_label(record) for record in records], dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    eligible_array = np.asarray([eligible(record) for record in records], dtype=bool)
    raw_vqa = np.asarray(
        [official_vqa_accuracy(record.generated_answer, record.references) for record in records],
        dtype=np.float64,
    )
    literal = np.asarray(
        [normalise_answer(record.generated_answer) == "unanswerable" for record in records],
        dtype=bool,
    )
    answerable = labels == 1
    unanswerable = labels == 0
    points: list[RiskCoveragePoint] = []
    for threshold in thresholds:
        accepted = eligible_array & (score_array >= threshold)
        accepted_accuracy = float(raw_vqa[accepted].mean()) if accepted.any() else None
        answerable_recall = float(accepted[answerable].mean()) if answerable.any() else 0.0
        unsafe = float(accepted[unanswerable].mean()) if unanswerable.any() else 0.0
        specific_unsafe = (
            float((accepted & unanswerable & ~literal).sum() / unanswerable.sum())
            if unanswerable.any()
            else 0.0
        )
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                coverage=float(accepted.mean()),
                accepted_vqa_accuracy=accepted_accuracy,
                selective_risk=1.0 - accepted_accuracy if accepted_accuracy is not None else None,
                answerable_recall=answerable_recall,
                unsafe_answer_rate=unsafe,
                specific_unsafe_answer_rate=specific_unsafe,
            )
        )
    return tuple(points)


def _sample_thresholds(scores: Sequence[float], maximum_points: int) -> tuple[float, ...]:
    """Retain deterministic, coverage-spanning thresholds including both endpoints."""
    values = np.unique(np.asarray(scores, dtype=np.float64))[::-1]
    if values.size > maximum_points - 2:
        indices = np.linspace(0, values.size - 1, maximum_points - 2, dtype=np.int64)
        values = values[np.unique(indices)]
    upper = float(np.nextafter(values[0], math.inf))
    lower = float(np.nextafter(values[-1], -math.inf))
    return tuple([upper, *values.tolist(), lower])


def _answerability_label(record: GenerationRecord) -> int:
    """Return a statically narrowed binary label from a labelled record."""
    if record.answerable is None:
        raise ValueError("policy metrics require answerability labels")
    return record.answerable


def _head_score(record: GenerationRecord) -> float:
    """Return a statically narrowed head score from a head-enabled record."""
    if record.answerability_score is None:
        raise ValueError("head policy requires answerability scores")
    return record.answerability_score


def _write_curve_csv(comparison: PolicyComparison, path: Path) -> None:
    """Write all curve points in long CSV format."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["policy", *RiskCoveragePoint.__dataclass_fields__.keys()],
        )
        writer.writeheader()
        for policy_name, points in comparison.curves.items():
            for point in points:
                writer.writerow({"policy": policy_name, **asdict(point)})


def _write_matched_recall_csv(comparison: PolicyComparison, path: Path) -> None:
    """Write paper-ready policy rows at predeclared matched-recall targets."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["policy", *MatchedRecallPoint.__dataclass_fields__.keys()],
        )
        writer.writeheader()
        for policy_name, points in comparison.matched_answerable_recall.items():
            for point in points:
                writer.writerow({"policy": policy_name, **asdict(point)})


def _write_curve_svg(comparison: PolicyComparison, path: Path) -> None:
    """Render a compact vector risk--coverage chart without plotting dependencies."""
    width, height = 760, 520
    left, right, top, bottom = 80, 30, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    colours = {
        "head_only": "#0072B2",
        "hybrid": "#009E73",
        "decoder_confidence": "#D55E00",
        "decoder_confidence_plus_literal": "#CC79A7",
    }
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" stroke="black"/>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        (
            f'<text x="{left + plot_width / 2}" y="{height - 20}" '
            'text-anchor="middle" font-family="sans-serif">Coverage</text>'
        ),
        (
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" '
            f'font-family="sans-serif" transform="rotate(-90 20 {top + plot_height / 2})">'
            "Selective risk (lower is better)</text>"
        ),
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot_width
        y = top + (1.0 - value) * plot_height
        elements.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" '
                    f'y2="{top + plot_height}" stroke="#dddddd"/>'
                ),
                (
                    f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" '
                    f'y2="{y:.1f}" stroke="#dddddd"/>'
                ),
                (
                    f'<text x="{x:.1f}" y="{top + plot_height + 25}" '
                    'text-anchor="middle" font-family="sans-serif" font-size="12">'
                    f"{value:.1f}</text>"
                ),
                (
                    f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="12">{value:.1f}</text>'
                ),
            ]
        )
    for index, (policy_name, points) in enumerate(comparison.curves.items()):
        coordinates = " ".join(
            (
                f"{left + point.coverage * plot_width:.1f},"
                f"{top + (1.0 - point.selective_risk) * plot_height:.1f}"
            )
            for point in points
            if point.selective_risk is not None
        )
        colour = colours.get(policy_name, "#333333")
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{colour}" stroke-width="2"/>'
        )
        legend_y = top + 18 * index
        elements.extend(
            [
                (
                    f'<line x1="{left + 10}" y1="{legend_y}" x2="{left + 35}" '
                    f'y2="{legend_y}" stroke="{colour}" stroke-width="3"/>'
                ),
                (
                    f'<text x="{left + 42}" y="{legend_y + 4}" '
                    f'font-family="sans-serif" font-size="12">{policy_name}</text>'
                ),
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _json_safe(value: object) -> object:
    """Replace non-finite metric leaves with JSON ``null`` recursively."""
    if isinstance(value, dict):
        return {name: _json_safe(child) for name, child in value.items()}
    if isinstance(value, list):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
