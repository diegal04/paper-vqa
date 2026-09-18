"""Prediction artefacts and result tables for paper-ready reporting."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One unconditional generation shared by all abstention policies."""

    sample_id: str
    generated_answer: str
    answerability_score: float | None
    decoder_score: float
    references: tuple[str, ...]
    answerable: int | None


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One auditable model prediction and its evaluation context."""

    sample_id: str
    prediction: str | None
    generated_answer: str
    answerability_score: float | None
    decoder_score: float
    accepted: bool
    references: tuple[str, ...]
    answerable: int | None


def write_predictions(records: Sequence[PredictionRecord], path: Path) -> None:
    """Write test predictions as JSON without omitting references or abstentions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_generation_records(path: Path) -> list[GenerationRecord]:
    """Reconstruct unconditional generations from a saved prediction report.

    Args:
        path: JSON file previously produced by :func:`write_predictions`.

    Returns:
        Validated generation records suitable for policy recalculation.

    Raises:
        ValueError: If the report does not contain the required typed fields.
    """
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("prediction report must contain a JSON list")
    return [_generation_record(item, index) for index, item in enumerate(payload)]


def _generation_record(payload: object, index: int) -> GenerationRecord:
    """Validate one serialized prediction and retain policy-relevant fields."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"prediction entry {index} must be an object")
    sample_id = _required_string(payload, "sample_id", index)
    generated_answer = _required_string(payload, "generated_answer", index)
    decoder_score = _required_float(payload, "decoder_score", index)
    answerability_score = _optional_float(payload.get("answerability_score"), index)
    references_value = payload.get("references")
    if not isinstance(references_value, list) or not references_value:
        raise ValueError(f"prediction entry {index} requires non-empty references")
    if not all(isinstance(reference, str) for reference in references_value):
        raise ValueError(f"prediction entry {index} references must be strings")
    answerable_value = payload.get("answerable")
    if answerable_value is not None and (
        not isinstance(answerable_value, int)
        or isinstance(answerable_value, bool)
        or answerable_value not in (0, 1)
    ):
        raise ValueError(f"prediction entry {index} answerable must be 0, 1, or null")
    return GenerationRecord(
        sample_id=sample_id,
        generated_answer=generated_answer,
        answerability_score=answerability_score,
        decoder_score=decoder_score,
        references=tuple(references_value),
        answerable=answerable_value,
    )


def _required_string(payload: Mapping[Any, Any], name: str, index: int) -> str:
    """Read one required string field from a prediction object."""
    value = payload.get(name)
    if not isinstance(value, str):
        raise ValueError(f"prediction entry {index} field {name} must be a string")
    return value


def _required_float(payload: Mapping[Any, Any], name: str, index: int) -> float:
    """Read one required finite numeric field from a prediction object."""
    value = payload.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"prediction entry {index} field {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"prediction entry {index} field {name} must be in [0, 1]")
    return result


def _optional_float(value: object, index: int) -> float | None:
    """Read an optional numeric answerability score."""
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"prediction entry {index} answerability_score must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"prediction entry {index} answerability_score must be in [0, 1]")
    return result
