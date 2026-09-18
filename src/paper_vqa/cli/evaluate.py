"""Hydra entry point for validation calibration and one frozen-test evaluation."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolve_device, resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.datamodule import VQADataModule
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
from paper_vqa.utils.reproducibility import seed_everything


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Load a selected checkpoint, calibrate on validation, and score test once."""
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
    processor = build_processor(str(values["model"]["name"]), values["model"].get("revision"))
    loaders = VQADataModule(
        values["data"], values["replay"], processor, trainer_config, values["loss"]
    ).build(include_test=True)
    if not loaders.test_examples:
        raise ValueError("frozen test evaluation requires a non-empty test partition")
    output_directory = Path(str(trainer_config["output_dir"])).parent
    write_manifests(loaders.manifests, output_directory)
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
    validation_generations = evaluator.generate_examples(
        loaders.validation_examples,
        dict(values["evaluation"]["generation"]),
    )
    thresholds = calibrate_policy_thresholds(
        validation_generations,
        float(values["evaluation"]["minimum_answerable_recall"]),
        [float(target) for target in values["evaluation"]["matched_answerable_recall_targets"]],
    )
    configured_threshold = values["evaluation"]["threshold"]
    head_threshold = (
        float(configured_threshold) if configured_threshold is not None else thresholds.head
    )
    if model.answerability_head is None:
        head_threshold = 0.0
    if head_threshold is None:
        raise RuntimeError("head-enabled evaluation did not produce a calibration threshold")
    if not 0.0 <= head_threshold <= 1.0:
        raise ValueError("evaluation.threshold must be in [0, 1]")
    thresholds = CalibratedThresholds(
        head=head_threshold if model.answerability_head is not None else None,
        decoder_confidence=thresholds.decoder_confidence,
        matched_answerable_recall=thresholds.matched_answerable_recall,
    )
    test_generations = evaluator.generate_examples(
        loaders.test_examples,
        dict(values["evaluation"]["generation"]),
    )
    result, predictions = evaluator.evaluate_records(
        test_generations, head_threshold, int(values["evaluation"]["ece_bins"])
    )
    comparison = compare_abstention_policies(
        test_generations,
        thresholds,
        int(values["evaluation"]["risk_coverage_points"]),
    )
    evaluation_directory = output_directory / "evaluation"
    write_evaluation(result, predictions, evaluation_directory)
    write_policy_comparison(comparison, evaluation_directory)
    tracker = build_tracker(values["logging"], values)
    try:
        tracker.log_artifact(output_directory / "manifests", "data-manifests", "dataset")
        tracker.log_artifact(output_directory / "evaluation", "frozen-test-results", "evaluation")
        metrics = flatten_evaluation_metrics(result, "test")
        metrics.update(flatten_policy_metrics(comparison, "test"))
        tracker.log(metrics, step=0)
    finally:
        tracker.finish()
    print(result)


if __name__ == "__main__":
    main()
