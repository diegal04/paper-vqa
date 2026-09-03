"""Factories that construct reproducible BLIP and PEFT model variants."""

from collections.abc import Mapping
from typing import Any, cast

from paper_vqa.models.answerability import AnswerabilityHead, PoolingStrategy
from paper_vqa.models.selective_vqa import SelectiveVQAModel


def build_vqa_model(
    model_config: Mapping[str, Any], head_config: Mapping[str, Any]
) -> SelectiveVQAModel:
    """Instantiate the configured BLIP, LoRA adapters and optional head.

    Args:
        model_config: BLIP/LoRA configuration.
        head_config: Auxiliary-head configuration.

    Returns:
        A fresh experiment model.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import BlipForQuestionAnswering

    revision = model_config.get("revision")
    blip: Any
    if revision:
        blip = BlipForQuestionAnswering.from_pretrained(
            model_config["name"], revision=str(revision)
        )
    else:
        blip = BlipForQuestionAnswering.from_pretrained(model_config["name"])
    lora = model_config["lora"]
    if bool(lora["enabled"]):
        blip = get_peft_model(
            blip,
            LoraConfig(
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                bias="none",
            ),
        )
    if not bool(head_config["enabled"]):
        return SelectiveVQAModel(blip, None)
    pooling = str(head_config["pooling"])
    if pooling not in {"masked_mean", "cls"}:
        raise ValueError(f"unsupported pooling strategy: {pooling}")
    hidden_size = int(blip.config.text_config.hidden_size)
    head = AnswerabilityHead(
        input_dim=hidden_size,
        hidden_dim=int(head_config["hidden_dim"]),
        dropout=float(head_config["dropout"]),
        pooling=cast(PoolingStrategy, pooling),
    )
    return SelectiveVQAModel(blip, head)


def build_processor(model_name: str, revision: str | None = None) -> Any:
    """Load the matching BLIP processor at an optional immutable revision.

    Args:
        model_name: Identifier of the BLIP checkpoint.
        revision: Optional immutable Hugging Face revision of that checkpoint.
    """
    from transformers import BlipProcessor

    return (
        BlipProcessor.from_pretrained(model_name, revision=revision)
        if revision
        else BlipProcessor.from_pretrained(model_name)
    )
