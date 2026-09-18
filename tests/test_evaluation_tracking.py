from paper_vqa.evaluation.metrics import AnswerabilityMetrics, SelectiveMetrics
from paper_vqa.evaluation.runner import EvaluationResult, flatten_evaluation_metrics


def test_flatten_evaluation_metrics_logs_every_available_scalar() -> None:
    """Nested result groups should become stable slash-delimited tracker keys."""
    result = EvaluationResult(
        threshold=0.4,
        vqa_accuracy=0.5,
        answerability=AnswerabilityMetrics(0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.45, 0.1, 0.05),
        selective=SelectiveMetrics(0.6, 0.7, 0.3, 0.2, 0.1, 0.55, 0.15),
    )

    metrics = flatten_evaluation_metrics(result, "validation")

    assert metrics["validation/threshold"] == 0.4
    assert metrics["validation/vqa_accuracy"] == 0.5
    assert metrics["validation/answerability/average_precision"] == 0.9
    assert metrics["validation/selective/specific_unsafe_answer_rate"] == 0.1


def test_flatten_evaluation_metrics_omits_unavailable_groups() -> None:
    """VQA-only baselines should not invent answerability metrics."""
    metrics = flatten_evaluation_metrics(EvaluationResult(0.0, 0.25, None, None), "validation/")

    assert metrics == {"validation/threshold": 0.0, "validation/vqa_accuracy": 0.25}
