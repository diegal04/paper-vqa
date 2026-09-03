import pytest

from paper_vqa.cli.baseline import _load_examples, _validate_zero_shot_configuration


def test_zero_shot_rejects_lora_or_head() -> None:
    model = {"lora": {"enabled": False}}
    head = {"enabled": False}

    _validate_zero_shot_configuration(model, head)
    with pytest.raises(ValueError, match="LoRA"):
        _validate_zero_shot_configuration({"lora": {"enabled": True}}, head)
    with pytest.raises(ValueError, match="head"):
        _validate_zero_shot_configuration(model, {"enabled": True})


def test_zero_shot_refuses_frozen_test() -> None:
    with pytest.raises(ValueError, match="frozen test"):
        _load_examples({"test": {}}, {"split": "test", "max_samples": None})
