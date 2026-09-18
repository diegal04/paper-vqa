"""Adapters that preserve VQA annotations and build model-ready batches."""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from paper_vqa.data.records import SourceConfig, TrainingTargetPolicy, VQAExample
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


@dataclass(frozen=True, slots=True)
class HuggingFaceImageReference:
    """Lazy pointer to one image in a Hugging Face dataset row."""

    dataset: Any
    index: int
    image_field: str

    def load(self) -> Any:
        """Decode the image only when a DataLoader requests this sample."""
        return self.dataset[self.index][self.image_field]


def load_rgb_image(value: Any) -> Image.Image:
    """Open paths and normalise a supported image object to RGB."""
    if isinstance(value, HuggingFaceImageReference):
        return load_rgb_image(value.load())
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
        rows = self._text_rows(dataset)
        examples = [
            example
            for index, row in enumerate(rows)
            if (example := self._to_example(self._with_lazy_image(row, dataset, index), index))
            is not None
        ]
        return self._sample(examples)

    def _text_rows(self, dataset: Any) -> Iterable[Mapping[str, Any]]:
        """Expose only metadata columns so manifest creation never decodes images."""
        fields = [self.config.question_field, self.config.answers_field]
        if self.config.answerable_field is not None:
            fields.append(self.config.answerable_field)
        if self.config.id_field is not None:
            fields.append(self.config.id_field)
        unique_fields = list(dict.fromkeys(fields))
        if hasattr(dataset, "column_names"):
            missing = set(unique_fields).difference(dataset.column_names)
            if missing:
                raise ValueError(f"hosted source is missing required columns: {sorted(missing)}")
        if hasattr(dataset, "select_columns"):
            return cast(Iterable[Mapping[str, Any]], dataset.select_columns(unique_fields))
        return ({field: row.get(field) for field in unique_fields} for row in dataset)

    def _with_lazy_image(self, row: Mapping[str, Any], dataset: Any, index: int) -> dict[str, Any]:
        """Attach a lazy image pointer to an otherwise text-only hosted row."""
        enriched = dict(row)
        enriched[self.config.image_field] = HuggingFaceImageReference(
            dataset=dataset,
            index=index,
            image_field=self.config.image_field,
        )
        return enriched


class HuggingFaceWithAnnotationsVQAAdapter(HuggingFaceVQAAdapter):
    """Join Hugging Face images/questions with official local benchmark annotations."""

    def load(self) -> list[VQAExample]:
        """Load rows and replace unlabelled hosted test fields with official labels.

        Raises:
            ValueError: If the hosted and official releases do not have identical IDs.
        """
        from datasets import load_dataset

        annotations = self._annotation_index()
        dataset = load_dataset(
            self.config.path,
            split=self.config.split,
            revision=self.config.revision,
        )
        source_ids: set[str] = set()
        examples: list[VQAExample] = []
        for index, raw_row in enumerate(self._text_rows(dataset)):
            row = self._with_lazy_image(raw_row, dataset, index)
            source_id = self._source_id(row, index)
            if source_id in source_ids:
                raise ValueError(f"duplicate hosted sample ID: {source_id}")
            source_ids.add(source_id)
            annotation = annotations.get(source_id)
            if annotation is None:
                raise ValueError(f"missing official annotation for hosted sample: {source_id}")
            official_question = str(annotation.get(self.config.question_field, "")).strip()
            if (
                official_question
                and official_question != str(row.get(self.config.question_field, "")).strip()
            ):
                raise ValueError(f"question mismatch for sample: {source_id}")
            row[self.config.answers_field] = annotation.get(self.config.answers_field, ())
            if self.config.answerable_field is not None:
                row[self.config.answerable_field] = annotation.get(self.config.answerable_field)
            example = self._to_example(row, index)
            if example is None:
                raise ValueError(f"incomplete official annotation for sample: {source_id}")
            examples.append(example)
        extra_annotations = set(annotations).difference(source_ids)
        if extra_annotations:
            raise ValueError(
                "official annotations contain "
                f"{len(extra_annotations)} IDs absent from hosted source"
            )
        return self._sample(examples)

    def _annotation_index(self) -> dict[str, Mapping[str, Any]]:
        """Read official JSON and index annotations by the configured image ID."""
        if self.config.annotations_path is None:
            raise ValueError("annotations_path is required")
        payload = json.loads(self.config.annotations_path.read_text(encoding="utf-8"))
        rows: Sequence[Mapping[str, Any]]
        if isinstance(payload, Mapping):
            rows = payload.get("annotations", payload.get("data", []))
        else:
            rows = payload
        index: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            identifier = str(row.get(self.config.annotation_id_field, ""))
            if not identifier:
                raise ValueError("official annotation has no image identifier")
            if identifier in index:
                raise ValueError(f"duplicate official annotation ID: {identifier}")
            index[identifier] = row
        if not index:
            raise ValueError("official annotation file has no records")
        return index

    def _source_id(self, row: Mapping[str, Any], position: int) -> str:
        """Extract the explicit hosted filename/ID used to join annotations."""
        if self.config.id_field is None:
            raise ValueError("hybrid adapter requires id_field")
        identifier = row.get(self.config.id_field)
        if identifier is None:
            raise ValueError(f"hosted row {position} has no {self.config.id_field!r}")
        return str(identifier)


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
    if config.backend == "huggingface_with_annotations":
        return HuggingFaceWithAnnotationsVQAAdapter(config)
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
        target_policy: TrainingTargetPolicy,
    ) -> None:
        """Store records, processor parameters, and decoder-target policy."""
        self.examples = tuple(examples)
        self.processor = processor
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.target_policy = target_policy

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
            example.target_answer(self.target_policy),
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
