import pytest

from paper_vqa.cli.evaluate_development import _load_validation_examples, _threshold


def test_development_evaluation_rejects_non_validation_split() -> None:
    with pytest.raises(ValueError, match="data.validation"):
        _load_validation_examples({"train": {}}, {"split": "train", "max_samples": None})


def test_development_threshold_uses_validation_calibration_by_default() -> None:
    assert _threshold({"threshold": None}, has_head=False) == 0.0
    assert _threshold({"threshold": None}, has_head=True) is None
    assert _threshold({"threshold": 0.75}, has_head=True) == 0.75
