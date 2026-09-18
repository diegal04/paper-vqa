"""Hydra entry point for an unadapted BLIP-VQA validation baseline."""

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolve_device, resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.datasets import build_adapter
from paper_vqa.data.records import SourceConfig, VQAExample
from paper_vqa.evaluation.policies import (
    calibrate_policy_thresholds,
    compare_abstention_policies,
    flatten_policy_metrics,
    write_policy_comparison,
)
from paper_vqa.evaluation.runner import Evaluator, flatten_evaluation_metrics, write_evaluation
from paper_vqa.models.factory import build_processor, build_vqa_model
from paper_vqa.training.tracker import build_tracker
from paper_vqa.utils.manifests import DatasetManifest
from paper_vqa.utils.reproducibility import seed_everything


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Evaluate frozen, unadapted BLIP-VQA on a non-test development split."""
    values = resolved_config(config)
    _validate_zero_shot_configuration(values["model"], values["head"])
    trainer_config = values["trainer"]
    trainer_config["device"] = resolve_device(str(trainer_config["device"]))
    seed_everything(
        int(trainer_config["seed"]),
        bool(trainer_config["deterministic"]),
        int(trainer_config["cpu_threads"]),
    )
    examples, manifest = _load_examples(values["data"], values["baseline"])
    output_directory = Path(str(trainer_config["output_dir"])).parent
    write_manifests((manifest,), output_directory)
    processor = build_processor(str(values["model"]["name"]), values["model"].get("revision"))
    model = build_vqa_model(values["model"], values["head"])
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
    result, predictions = evaluator.evaluate_records(
        generations,
        float(values["baseline"]["threshold"]),
        int(values["evaluation"]["ece_bins"]),
    )
    comparison = compare_abstention_policies(
        generations,
        thresholds,
        int(values["evaluation"]["risk_coverage_points"]),
    )
    baseline_directory = output_directory / "baseline"
    write_evaluation(result, predictions, baseline_directory)
    write_policy_comparison(comparison, baseline_directory)
    tracker = build_tracker(values["logging"], values)
    try:
        tracker.log_artifact(output_directory / "manifests", "data-manifests", "dataset")
        tracker.log_artifact(baseline_directory, "zero-shot-baseline", "evaluation")
        metrics = flatten_evaluation_metrics(result, "validation")
        metrics.update(flatten_policy_metrics(comparison, "validation"))
        tracker.log(metrics, step=0)
    finally:
        tracker.finish()
    print(result)


def _validate_zero_shot_configuration(
    model_config: dict[str, Any], head_config: dict[str, Any]
) -> None:
    """Reject accidental adaptation in the explicitly zero-shot baseline.

    Args:
        model_config: Resolved BLIP and LoRA configuration.
        head_config: Resolved auxiliary-head configuration.
    """
    if bool(model_config["lora"]["enabled"]):
        raise ValueError("zero-shot baseline requires LoRA to be disabled")
    if bool(head_config["enabled"]):
        raise ValueError("zero-shot baseline requires head.enabled=false")


def _load_examples(
    data_config: dict[str, Any], baseline_config: dict[str, Any]
) -> tuple[list[VQAExample], DatasetManifest]:
    """Load the configured development partition and its immutable manifest.

    Args:
        data_config: Resolved primary-source configuration.
        baseline_config: Split and deterministic sample-limit configuration.

    Returns:
        Evaluation examples and their source manifest.
    """
    split = str(baseline_config["split"])
    if split == "test":
        raise ValueError("zero-shot baseline must not evaluate frozen test")
    if split not in data_config:
        raise ValueError(f"baseline split is not configured: {split}")
    source = dict(data_config[split])
    max_samples = baseline_config.get("max_samples")
    if max_samples is not None:
        source["max_samples"] = int(max_samples)
    return build_adapter(SourceConfig.from_mapping(source)).manifest()


if __name__ == "__main__":
    main()
