"""Hydra entry point for checkpoint evaluation on VizWiz validation only."""

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolve_device, resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.datasets import build_adapter
from paper_vqa.data.records import SourceConfig, VQAExample
from paper_vqa.evaluation.policies import (
    CalibratedThresholds,
    calibrate_policy_thresholds,
    compare_abstention_policies,
    flatten_policy_metrics,
    write_policy_comparison,
)
from paper_vqa.evaluation.runner import Evaluator, flatten_evaluation_metrics, write_evaluation
from paper_vqa.models.factory import build_processor, build_vqa_model
from paper_vqa.training.checkpoints import CheckpointManager
from paper_vqa.training.tracker import build_tracker
from paper_vqa.utils.manifests import DatasetManifest
from paper_vqa.utils.reproducibility import seed_everything


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Score a selected checkpoint on validation without opening frozen test."""
    values = resolved_config(config)
    trainer_config = values["trainer"]
    trainer_config["device"] = resolve_device(str(trainer_config["device"]))
    checkpoint_path = values["evaluation"]["checkpoint_path"]
    if not checkpoint_path:
        raise ValueError("Set evaluation.checkpoint_path to a checkpoint directory")
    seed_everything(
        int(trainer_config["seed"]),
        bool(trainer_config["deterministic"]),
        int(trainer_config["cpu_threads"]),
    )
    examples, manifest = _load_validation_examples(values["data"], values["development"])
    output_directory = Path(str(trainer_config["output_dir"])).parent
    write_manifests((manifest,), output_directory)
    processor = build_processor(str(values["model"]["name"]), values["model"].get("revision"))
    model = build_vqa_model(values["model"], values["head"])
    CheckpointManager(Path(str(checkpoint_path))).load(model)
    progress = trainer_config.get("progress", {})
    evaluator = Evaluator(
        model,
        processor,
        str(trainer_config["device"]),
        progress_enabled=bool(progress.get("enabled", True)),
        progress_leave=bool(progress.get("leave", False)),
    )
    generations = evaluator.generate_examples(
        examples,
        dict(values["evaluation"]["generation"]),
    )
    thresholds = calibrate_policy_thresholds(
        generations,
        float(values["evaluation"]["minimum_answerable_recall"]),
        [float(target) for target in values["evaluation"]["matched_answerable_recall_targets"]],
    )
    configured_threshold = _threshold(values["development"], model.answerability_head is not None)
    threshold = thresholds.head if configured_threshold is None else configured_threshold
    if threshold is None:
        raise RuntimeError("head-enabled evaluation did not produce a calibration threshold")
    thresholds = CalibratedThresholds(
        head=threshold if model.answerability_head is not None else None,
        decoder_confidence=thresholds.decoder_confidence,
        matched_answerable_recall=thresholds.matched_answerable_recall,
    )
    result, predictions = evaluator.evaluate_records(
        generations, threshold, int(values["evaluation"]["ece_bins"])
    )
    comparison = compare_abstention_policies(
        generations,
        thresholds,
        int(values["evaluation"]["risk_coverage_points"]),
    )
    evaluation_directory = output_directory / "development_evaluation"
    write_evaluation(result, predictions, evaluation_directory)
    write_policy_comparison(comparison, evaluation_directory)
    tracker = build_tracker(values["logging"], values)
    try:
        tracker.log_artifact(output_directory / "manifests", "data-manifests", "dataset")
        tracker.log_artifact(evaluation_directory, "validation-results", "evaluation")
        metrics = flatten_evaluation_metrics(result, "validation")
        metrics.update(flatten_policy_metrics(comparison, "validation"))
        tracker.log(metrics, step=0)
    finally:
        tracker.finish()
    print(result)


def _load_validation_examples(
    data_config: dict[str, Any], development_config: dict[str, Any]
) -> tuple[list[VQAExample], DatasetManifest]:
    """Load the configured validation partition and its immutable manifest.

    Args:
        data_config: Resolved primary source configuration.
        development_config: Development split and deterministic sample limit.

    Returns:
        Validation examples and the associated source manifest.
    """
    split = str(development_config["split"])
    if split != "validation":
        raise ValueError("development evaluation is restricted to data.validation")
    source = dict(data_config[split])
    max_samples = development_config.get("max_samples")
    if max_samples is not None:
        source["max_samples"] = int(max_samples)
    return build_adapter(SourceConfig.from_mapping(source)).manifest()


def _threshold(development_config: dict[str, Any], has_head: bool) -> float | None:
    """Resolve a configured threshold or request validation-only calibration.

    Args:
        development_config: Configured fixed threshold, if any.
        has_head: Whether the checkpoint contains an answerability head.

    Returns:
        Zero for VQA-only models, a configured fixed threshold, or ``None``
        when an answerability head must be calibrated on validation.
    """
    if not has_head:
        return 0.0
    value = development_config.get("threshold")
    if value is None:
        return None
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("development.threshold must be in [0, 1]")
    return threshold


if __name__ == "__main__":
    main()
