import json
from dataclasses import asdict
from pathlib import Path

from paper_vqa.evaluation.policies import (
    CalibratedThresholds,
    MatchedRecallThresholds,
    calibrate_policy_thresholds,
    compare_abstention_policies,
    write_policy_comparison,
)
from paper_vqa.evaluation.reporting import GenerationRecord, read_generation_records


def _records() -> list[GenerationRecord]:
    return [
        GenerationRecord("a1", "yes", 0.9, 0.9, ("yes",) * 10, 1),
        GenerationRecord("a2", "unanswerable", 0.8, 0.2, ("blue",) * 10, 1),
        GenerationRecord("u1", "unanswerable", 0.7, 0.8, ("unanswerable",) * 10, 0),
        GenerationRecord("u2", "red", 0.1, 0.1, ("unanswerable",) * 10, 0),
    ]


def test_policy_calibration_uses_each_confidence_signal_independently() -> None:
    thresholds = calibrate_policy_thresholds(
        _records(),
        minimum_answerable_recall=1.0,
        matched_answerable_recall_targets=(0.5, 0.75),
    )

    assert thresholds.head == 0.8
    assert thresholds.decoder_confidence == 0.2
    assert thresholds.matched_answerable_recall[0] == MatchedRecallThresholds(0.5, 0.9, 0.9)
    assert thresholds.matched_answerable_recall[1] == MatchedRecallThresholds(0.75, None, None)


def test_policy_comparison_isolates_literal_head_and_hybrid_decisions() -> None:
    comparison = compare_abstention_policies(
        _records(),
        CalibratedThresholds(
            head=0.5,
            decoder_confidence=0.5,
            matched_answerable_recall=(MatchedRecallThresholds(0.5, 0.9, 0.9),),
        ),
        8,
    )

    policies = comparison.policies
    assert policies["always_answer"].selective.coverage == 1.0
    assert policies["always_answer"].selective.specific_unsafe_answer_rate == 0.5
    assert policies["decoder_literal"].selective.coverage == 0.5
    assert policies["decoder_confidence"].selective.specific_unsafe_answer_rate == 0.0
    assert policies["decoder_confidence_plus_literal"].selective.coverage == 0.25
    assert policies["head_only"].selective.coverage == 0.75
    assert policies["hybrid"].selective.coverage == 0.25
    assert policies["hybrid"].selective.specific_unsafe_answer_rate == 0.0
    assert set(comparison.curves) == {
        "decoder_confidence",
        "decoder_confidence_plus_literal",
        "head_only",
        "hybrid",
    }
    assert set(comparison.matched_answerable_recall) == {
        "decoder_confidence_plus_literal",
        "hybrid",
    }
    decoder_point = comparison.matched_answerable_recall["decoder_confidence_plus_literal"][0]
    assert decoder_point.target_answerable_recall == 0.5
    assert decoder_point.achieved_answerable_recall == 0.5
    assert decoder_point.specific_unsafe_count == 0


def test_policy_report_writes_json_csv_and_vector_plot(tmp_path: Path) -> None:
    comparison = compare_abstention_policies(
        _records(), CalibratedThresholds(head=1.0, decoder_confidence=1.0), 8
    )

    write_policy_comparison(comparison, tmp_path)

    payload = json.loads((tmp_path / "policy_metrics.json").read_text(encoding="utf-8"))
    assert "hybrid" in payload["policies"]
    assert "decoder_confidence_plus_literal" in payload["policies"]
    assert payload["policies"]["head_only"]["selective"]["accepted_vqa_accuracy"] is None
    assert "matched_answerable_recall" in payload
    assert (tmp_path / "risk_coverage.csv").read_text(encoding="utf-8").startswith("policy,")
    assert (
        (tmp_path / "matched_answerable_recall.csv")
        .read_text(encoding="utf-8")
        .startswith("policy,")
    )
    assert "<svg" in (tmp_path / "risk_coverage.svg").read_text(encoding="utf-8")


def test_saved_predictions_can_rebuild_generation_records(tmp_path: Path) -> None:
    records = _records()
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps([asdict(record) for record in records]), encoding="utf-8")

    assert read_generation_records(path) == records
