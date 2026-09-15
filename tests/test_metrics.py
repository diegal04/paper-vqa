import math

from paper_vqa.evaluation.metrics import (
    official_vqa_accuracy,
    select_safety_threshold,
    selective_metrics,
)


def test_official_vqa_accuracy_uses_leave_one_out_protocol() -> None:
    references = ["yes"] + ["no"] * 9

    assert official_vqa_accuracy("yes", references) == 0.3
    assert official_vqa_accuracy("no", references) == 1.0


def test_safety_threshold_respects_answerable_recall() -> None:
    result = select_safety_threshold([1, 1, 0, 0], [0.9, 0.7, 0.6, 0.1], 1.0)

    assert result.threshold <= 0.7
    assert result.answerable_recall == 1.0
    assert result.unsafe_answer_rate == 0.0


def test_selective_metrics_report_unsafe_responses() -> None:
    metrics = selective_metrics(
        [True, False, True, True],
        [1.0, 0.0, 0.5, 0.0],
        [1, 0, 0, 1],
        ["yes", None, "Unanswerable.", "no"],
    )

    assert metrics.coverage == 3 / 4
    assert metrics.unsafe_answer_rate == 0.5
    assert metrics.answerable_vqa_accuracy == 0.5
    assert metrics.accepted_unanswerable_answer_rate == 1 / 3
    assert not math.isnan(metrics.selective_risk)
