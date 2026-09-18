import math

from paper_vqa.evaluation.metrics import (
    normalise_answer,
    official_vqa_accuracy,
    select_safety_threshold,
    selective_metrics,
)


def test_official_vqa_accuracy_uses_leave_one_out_protocol() -> None:
    references = ["yes"] + ["no"] * 9

    assert official_vqa_accuracy("yes", references) == 0.3
    assert official_vqa_accuracy("no", references) == 1.0


def test_vqa_normalisation_matches_official_numbers_articles_and_punctuation() -> None:
    assert normalise_answer("The two cats!") == "2 cats"
    assert normalise_answer("1,000") == "1000"
    assert normalise_answer("3.14") == "3.14"
    assert normalise_answer("dont") == "don't"


def test_official_vqa_accuracy_preserves_unanimous_reference_special_case() -> None:
    references = ["Two"] * 10

    assert official_vqa_accuracy("Two", references) == 1.0
    assert official_vqa_accuracy("two", references) == 0.0


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
    assert metrics.specific_unsafe_answer_rate == 0.0
    assert metrics.answerable_vqa_accuracy == 0.5
    assert metrics.accepted_unanswerable_answer_rate == 1 / 3
    assert not math.isnan(metrics.selective_risk)


def test_specific_unsafe_rate_excludes_textual_abstentions() -> None:
    metrics = selective_metrics(
        [True, True, True, False],
        [0.0, 0.0, 0.0, 0.0],
        [0, 0, 1, 0],
        ["unanswerable", "blue", "yes", None],
    )

    assert metrics.unsafe_answer_rate == 2 / 3
    assert metrics.specific_unsafe_answer_rate == 1 / 3
