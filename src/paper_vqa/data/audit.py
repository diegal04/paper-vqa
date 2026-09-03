"""Reproducible descriptive audits for VQA dataset partitions."""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median

import torch

from paper_vqa.data.datasets import load_rgb_image
from paper_vqa.data.records import VQAExample


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Descriptive statistics for a non-empty numerical sample."""

    minimum: float
    mean: float
    median: float
    maximum: float


@dataclass(frozen=True, slots=True)
class AnswerabilitySummary:
    """Counts and prevalence of available answerability labels."""

    labelled: int
    answerable: int
    unanswerable: int
    unlabelled: int
    answerable_rate: float | None


@dataclass(frozen=True, slots=True)
class ImageAudit:
    """Result of decoding a deterministic sample of partition images."""

    sampled: int
    modes: dict[str, int]
    widths: NumericSummary | None
    heights: NumericSummary | None


@dataclass(frozen=True, slots=True)
class PreviewExample:
    """Small human-readable record included in an audit report."""

    sample_id: str
    question: str
    answers: tuple[str, ...]
    answerable: int | None


@dataclass(frozen=True, slots=True)
class PartitionAudit:
    """Descriptive report for one immutable VQA partition."""

    dataset: str
    split: str
    examples: int
    unique_sample_ids: int
    unique_questions: int
    question_words: NumericSummary
    reference_answers: NumericSummary
    answerability: AnswerabilitySummary
    top_modal_answers: tuple[tuple[str, int], ...]
    image_audit: ImageAudit
    previews: tuple[PreviewExample, ...]


@dataclass(frozen=True, slots=True)
class DatasetAuditReport:
    """Serializable collection of VQA partition audits."""

    seed: int
    partitions: tuple[PartitionAudit, ...]

    def save(self, path: Path) -> None:
        """Write a deterministic JSON audit report.

        Args:
            path: Destination JSON path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


class VQADatasetAuditor:
    """Compute deterministic descriptive audits without model inference."""

    def __init__(self, preview_count: int, image_sample_count: int, top_answer_count: int) -> None:
        """Validate and store audit output limits.

        Args:
            preview_count: Number of text examples retained for manual inspection.
            image_sample_count: Number of images decoded to verify accessibility.
            top_answer_count: Number of frequent modal answers included in the report.
        """
        if min(preview_count, image_sample_count, top_answer_count) < 0:
            raise ValueError("audit counts must be non-negative")
        self.preview_count = preview_count
        self.image_sample_count = image_sample_count
        self.top_answer_count = top_answer_count

    def audit(self, examples: Sequence[VQAExample], seed: int) -> PartitionAudit:
        """Audit one non-empty partition using a partition-specific deterministic seed.

        Args:
            examples: Records with all reference answers preserved.
            seed: Random seed controlling only audit sample selection.

        Returns:
            A serializable summary of data integrity and annotation distributions.
        """
        if not examples:
            raise ValueError("cannot audit an empty partition")
        question_lengths = [len(example.question.split()) for example in examples]
        answer_counts = [len(example.answers) for example in examples]
        labels = [example.answerable for example in examples]
        labelled = [label for label in labels if label is not None]
        answerable = sum(label == 1 for label in labelled)
        modal_answers = Counter(example.training_answer for example in examples)
        sampled = self._sample(examples, max(self.preview_count, self.image_sample_count), seed)
        previews = tuple(
            PreviewExample(example.sample_id, example.question, example.answers, example.answerable)
            for example in sampled[: self.preview_count]
        )
        dataset_names = {example.dataset for example in examples}
        split_names = {example.split for example in examples}
        if len(dataset_names) != 1 or len(split_names) != 1:
            raise ValueError("an audit partition must contain exactly one dataset and split")
        return PartitionAudit(
            dataset=next(iter(dataset_names)),
            split=next(iter(split_names)),
            examples=len(examples),
            unique_sample_ids=len({example.sample_id for example in examples}),
            unique_questions=len({example.question for example in examples}),
            question_words=_summary(question_lengths),
            reference_answers=_summary(answer_counts),
            answerability=AnswerabilitySummary(
                labelled=len(labelled),
                answerable=answerable,
                unanswerable=len(labelled) - answerable,
                unlabelled=len(labels) - len(labelled),
                answerable_rate=answerable / len(labelled) if labelled else None,
            ),
            top_modal_answers=tuple(modal_answers.most_common(self.top_answer_count)),
            image_audit=self._audit_images(sampled[: self.image_sample_count]),
            previews=previews,
        )

    def _sample(
        self, examples: Sequence[VQAExample], count: int, seed: int
    ) -> tuple[VQAExample, ...]:
        """Draw a deterministic sample without replacement in output order.

        Args:
            examples: Candidate examples.
            count: Maximum number to select.
            seed: Seed for the local torch generator.

        Returns:
            Selected records, sorted by their original partition position.
        """
        if count == 0:
            return ()
        selected_count = min(count, len(examples))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(examples), generator=generator)[:selected_count].tolist()
        return tuple(examples[index] for index in sorted(indices))

    def _audit_images(self, examples: Sequence[VQAExample]) -> ImageAudit:
        """Decode selected images and summarise their RGB dimensions.

        Args:
            examples: Deterministically selected records whose images will be opened.

        Returns:
            Decoding count, RGB modes, and dimension distributions.
        """
        modes: Counter[str] = Counter()
        widths: list[int] = []
        heights: list[int] = []
        for example in examples:
            image = load_rgb_image(example.image)
            modes[image.mode] += 1
            widths.append(image.width)
            heights.append(image.height)
        return ImageAudit(
            sampled=len(examples),
            modes=dict(sorted(modes.items())),
            widths=_summary(widths) if widths else None,
            heights=_summary(heights) if heights else None,
        )


def _summary(values: Sequence[int]) -> NumericSummary:
    """Calculate numerical summary statistics for non-empty integer values.

    Args:
        values: Observed numeric values.

    Returns:
        Minimum, arithmetic mean, median, and maximum.
    """
    if not values:
        raise ValueError("cannot summarise an empty sequence")
    return NumericSummary(
        minimum=float(min(values)),
        mean=float(fmean(values)),
        median=float(median(values)),
        maximum=float(max(values)),
    )
