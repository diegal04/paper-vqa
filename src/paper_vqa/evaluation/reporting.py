"""Prediction artefacts and result tables for paper-ready reporting."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One auditable model prediction and its evaluation context."""

    sample_id: str
    prediction: str | None
    answerable_score: float
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
