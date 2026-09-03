"""Typed records shared by all dataset adapters."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


BackendName = Literal["huggingface", "local_json"]


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Configuration required to load one immutable dataset partition."""

    name: str
    backend: BackendName
    path: str
    split: str
    revision: str | None
    image_root: Path | None
    max_samples: int | None
    seed: int
    weight: float
    question_field: str = "question"
    answers_field: str = "answers"
    answerable_field: str | None = "answerable"
    image_field: str = "image"
    id_field: str | None = None

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SourceConfig":
        """Build a typed source configuration from a Hydra mapping.

        Args:
            mapping: Resolved configuration of one source.

        Returns:
            A validated source configuration.
        """
        max_samples = mapping.get("max_samples")
        if max_samples is not None and int(max_samples) < 1:
            raise ValueError("max_samples must be positive or null")
        image_root = mapping.get("image_root")
        weight = float(mapping.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("source weight must be positive")
        backend = str(mapping["backend"])
        if backend not in {"huggingface", "local_json"}:
            raise ValueError(f"unsupported backend: {backend}")
        return cls(
            name=str(mapping["name"]),
            backend=cast(BackendName, backend),
            path=str(mapping["path"]),
            split=str(mapping["split"]),
            revision=mapping.get("revision"),
            image_root=Path(image_root) if image_root else None,
            max_samples=int(max_samples) if max_samples is not None else None,
            seed=int(mapping["seed"]),
            weight=weight,
            question_field=str(mapping.get("question_field", "question")),
            answers_field=str(mapping.get("answers_field", "answers")),
            answerable_field=mapping.get("answerable_field", "answerable"),
            image_field=str(mapping.get("image_field", "image")),
            id_field=mapping.get("id_field"),
        )


@dataclass(frozen=True, slots=True)
class VQAExample:
    """One VQA sample preserving every available human reference answer."""

    sample_id: str
    dataset: str
    split: str
    image: Any
    question: str
    answers: tuple[str, ...]
    answerable: int | None

    def __post_init__(self) -> None:
        """Validate binary labels and non-empty question/answer fields."""
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not self.question.strip():
            raise ValueError("question cannot be empty")
        if not self.answers:
            raise ValueError("at least one reference answer is required")
        if self.answerable not in (None, 0, 1):
            raise ValueError("answerable must be 0, 1, or None")

    @property
    def has_answerable(self) -> bool:
        """Whether this example carries a genuine answerability annotation."""
        return self.answerable is not None

    @property
    def training_answer(self) -> str:
        """Return the deterministic modal answer used only for generation loss."""
        counts: dict[str, int] = {}
        for answer in self.answers:
            counts[answer] = counts.get(answer, 0) + 1
        highest_count = max(counts.values())
        return next(answer for answer in self.answers if counts[answer] == highest_count)
