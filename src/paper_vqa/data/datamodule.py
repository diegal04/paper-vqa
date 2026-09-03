"""Data-module orchestration with explicit leakage checks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from paper_vqa.data.datasets import RecordVQADataset, build_adapter
from paper_vqa.data.records import SourceConfig, VQAExample
from paper_vqa.utils.manifests import DatasetManifest, assert_disjoint_manifests
from paper_vqa.utils.reproducibility import make_generator, make_worker_init_fn


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Training and validation loaders together with their source records."""

    train: DataLoader[dict[str, Any]]
    validation: DataLoader[dict[str, Any]]
    train_examples: tuple[VQAExample, ...]
    validation_examples: tuple[VQAExample, ...]
    test_examples: tuple[VQAExample, ...]
    manifests: tuple[DatasetManifest, ...]


class VQADataModule:
    """Build loaders from Hydra sources while enforcing split disjointness."""

    def __init__(
        self,
        data_config: Mapping[str, Any],
        replay_config: Mapping[str, Any],
        processor: Any,
        trainer_config: Mapping[str, Any],
    ) -> None:
        """Store resolved configuration and model processor.

        Args:
            data_config: Primary VizWiz source mappings.
            replay_config: Zero or more additional training source mappings.
            processor: BLIP processor used to tokenise samples.
            trainer_config: Loader and token-length settings.
        """
        self.data_config = data_config
        self.replay_config = replay_config
        self.processor = processor
        self.trainer_config = trainer_config

    def build(self) -> DataLoaders:
        """Load all configured partitions and create deterministic loaders."""
        train_examples, train_manifest = self._load_source(self.data_config["train"])
        validation_examples, validation_manifest = self._load_source(self.data_config["validation"])
        test_examples, test_manifest = self._load_source(self.data_config["test"])
        replay_examples: list[VQAExample] = []
        replay_manifests: list[DatasetManifest] = []
        source_sizes_and_weights: list[tuple[int, float]] = [
            (len(train_examples), float(self.data_config["train"].get("weight", 1.0)))
        ]
        if bool(self.replay_config.get("enabled", False)):
            for source in self.replay_config.get("sources", []):
                examples, manifest = self._load_source(source)
                replay_examples.extend(examples)
                replay_manifests.append(manifest)
                source_sizes_and_weights.append((len(examples), float(source.get("weight", 1.0))))
        assert_disjoint_manifests(
            train_manifest,
            validation_manifest,
            test_manifest,
            *replay_manifests,
        )
        all_train = tuple(train_examples + replay_examples)
        train_dataset = self._dataset(all_train)
        validation_dataset = self._dataset(validation_examples)
        batch_size = int(self.trainer_config["batch_size"])
        seed = int(self.trainer_config["seed"])
        workers = int(self.trainer_config["num_workers"])
        sampler = self._build_sampler(source_sizes_and_weights, seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            generator=make_generator(seed),
            num_workers=workers,
            pin_memory=bool(self.trainer_config.get("pin_memory", True)),
            worker_init_fn=make_worker_init_fn(seed),
            persistent_workers=workers > 0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            generator=make_generator(seed + 1),
            num_workers=workers,
            pin_memory=bool(self.trainer_config.get("pin_memory", True)),
            worker_init_fn=make_worker_init_fn(seed),
            persistent_workers=workers > 0,
        )
        return DataLoaders(
            train=train_loader,
            validation=validation_loader,
            train_examples=all_train,
            validation_examples=tuple(validation_examples),
            test_examples=tuple(test_examples),
            manifests=(train_manifest, validation_manifest, test_manifest, *replay_manifests),
        )

    def _load_source(self, mapping: Mapping[str, Any]) -> tuple[list[VQAExample], DatasetManifest]:
        """Load one mapping through its selected adapter."""
        config = SourceConfig.from_mapping(dict(mapping))
        examples, manifest = build_adapter(config).manifest()
        return examples, manifest

    def _dataset(self, examples: Sequence[VQAExample]) -> Dataset[dict[str, Any]]:
        """Create a processor-backed dataset for one split."""
        return RecordVQADataset(
            examples=examples,
            processor=self.processor,
            max_question_length=int(self.trainer_config["max_question_length"]),
            max_answer_length=int(self.trainer_config["max_answer_length"]),
        )

    def _build_sampler(
        self, source_sizes_and_weights: Sequence[tuple[int, float]], seed: int
    ) -> WeightedRandomSampler | None:
        """Build optional source-level weighted replay sampling from YAML ratios."""
        if str(self.replay_config.get("sampling", "proportional")) != "source_weighted":
            return None
        weights: list[float] = []
        for source_size, source_weight in source_sizes_and_weights:
            if source_size < 1:
                continue
            weights.extend([source_weight / source_size] * source_size)
        if not weights:
            raise ValueError("weighted sampling requires at least one source sample")
        total = self.replay_config.get("samples_per_epoch") or len(weights)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=int(total),
            replacement=True,
            generator=make_generator(seed),
        )


def concatenate_datasets(datasets: Sequence[Dataset[dict[str, Any]]]) -> Dataset[dict[str, Any]]:
    """Concatenate datasets while rejecting an empty experiment definition."""
    if not datasets:
        raise ValueError("at least one dataset is required")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
