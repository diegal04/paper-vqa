"""Adapters that preserve VQA annotations and build model-ready batches."""

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from paper_vqa.data.records import SourceConfig, VQAExample
from paper_vqa.utils.manifests import DatasetManifest, ManifestEntry, manifest_from_entries


def _answer_texts(value: Any) -> tuple[str, ...]:
    """Normalise common answer-list schemas without discarding annotations."""
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Sequence):
        return ()
    answers: list[str] = []
    for item in value:
        candidate = item.get("answer", "") if isinstance(item, Mapping) else str(item)
        if str(candidate).strip():
            answers.append(str(candidate))
    return tuple(answers)


def load_rgb_image(value: Any) -> Image.Image:
    """Open paths and normalise a supported image object to RGB."""
    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            return opened.convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "convert"):
        return cast(Image.Image, value.convert("RGB"))
    raise TypeError(f"Unsupported image type: {type(value)!r}")


class VQAAdapter:
    """Base class for deterministic source adapters."""

    def __init__(self, config: SourceConfig) -> None:
        """Store immutable source configuration."""
        self.config = config

    def load(self) -> list[VQAExample]:
        """Load source records in their deterministic experiment order."""
        raise NotImplementedError

    def manifest(self) -> tuple[list[VQAExample], DatasetManifest]:
        """Load records and return their provenance manifest."""
        examples = self.load()
        manifest = manifest_from_entries(
            name=f"{self.config.name}:{self.config.split}",
            revision=self.config.revision,
            seed=self.config.seed,
            entries=(
                ManifestEntry(example.dataset, example.split, example.sample_id)
                for example in examples
            ),
        )
        return examples, manifest

    def _to_example(self, row: Mapping[str, Any], index: int) -> VQAExample | None:
        """Convert a raw row to a record or skip incomplete annotations."""
        question = str(row.get(self.config.question_field, "")).strip()
        answers = _answer_texts(row.get(self.config.answers_field, ()))
        if not question or not answers:
            return None
        raw_id = row.get(self.config.id_field) if self.config.id_field else None
        sample_id = str(raw_id or row.get("image_id") or row.get("image") or index)
        raw_label = row.get(self.config.answerable_field) if self.config.answerable_field else None
        answerable = int(raw_label) if raw_label is not None else None
        return VQAExample(
            sample_id=sample_id,
            dataset=self.config.name,
            split=self.config.split,
            image=row.get(self.config.image_field),
            question=question,
            answers=answers,
            answerable=answerable,
        )

    def _sample(self, examples: list[VQAExample]) -> list[VQAExample]:
        """Take a deterministic configured subset without hidden global state."""
        if self.config.max_samples is None or len(examples) <= self.config.max_samples:
            return examples
        generator = torch.Generator().manual_seed(self.config.seed)
        indices = torch.randperm(len(examples), generator=generator).tolist()
        return [examples[index] for index in indices[: self.config.max_samples]]


class HuggingFaceVQAAdapter(VQAAdapter):
    """Load an explicit Hugging Face split with deterministic row sampling."""

    def load(self) -> list[VQAExample]:
        """Load and validate a non-streaming dataset split.

        Returns:
            Valid examples, with all references preserved.
        """
        from datasets import load_dataset

        dataset = load_dataset(
            self.config.path,
            split=self.config.split,
            revision=self.config.revision,
        )
        rows: Iterable[Mapping[str, Any]] = dataset
        examples = [
            example
            for index, row in enumerate(rows)
            if (example := self._to_example(row, index)) is not None
        ]
        return self._sample(examples)


class LocalJsonVQAAdapter(VQAAdapter):
    """Load official JSON annotations and resolve images from a local directory."""

    def load(self) -> list[VQAExample]:
        """Load local JSON annotations distributed by a benchmark provider."""
        payload = json.loads(Path(self.config.path).read_text(encoding="utf-8"))
        rows: Sequence[Mapping[str, Any]]
        if isinstance(payload, Mapping):
            rows = payload.get("annotations", payload.get("data", []))
        else:
            rows = payload
        examples: list[VQAExample] = []
        for index, row in enumerate(rows):
            example = self._to_example(row, index)
            if example is None:
                continue
            image = example.image
            if self.config.image_root is not None and isinstance(image, str):
                image = self.config.image_root / image
            examples.append(
                VQAExample(
                    sample_id=example.sample_id,
                    dataset=example.dataset,
                    split=example.split,
                    image=image,
                    question=example.question,
                    answers=example.answers,
                    answerable=example.answerable,
                )
            )
        return self._sample(examples)


def build_adapter(config: SourceConfig) -> VQAAdapter:
    """Construct the adapter selected by configuration.

    Args:
        config: Typed source configuration.

    Returns:
        An adapter for its declared backend.
    """
    if config.backend == "huggingface":
        return HuggingFaceVQAAdapter(config)
    if config.backend == "local_json":
        return LocalJsonVQAAdapter(config)
    raise ValueError(f"Unsupported dataset backend: {config.backend}")


class RecordVQADataset(Dataset[dict[str, Tensor]]):
    """Tokenise typed VQA examples for a Hugging Face BLIP processor."""

    def __init__(
        self,
        examples: Sequence[VQAExample],
        processor: Any,
        max_question_length: int,
        max_answer_length: int,
    ) -> None:
        """Store records and processor parameters without mutating annotations."""
        self.examples = tuple(examples)
        self.processor = processor
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Process one image-question-answer training record."""
        example = self.examples[index]
        image = load_rgb_image(example.image)
        inputs = self.processor(
            images=image,
            text=example.question,
            return_tensors="pt",
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
        )
        answer_tokens = self.processor.tokenizer(
            example.training_answer,
            return_tensors="pt",
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
        )
        labels = answer_tokens.input_ids.squeeze(0)
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels,
            "answerable": torch.tensor(example.answerable or 0, dtype=torch.long),
            "has_answerable": torch.tensor(example.has_answerable, dtype=torch.bool),
        }
