"""Rebuild validation policy reports from previously saved generations."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolved_config
from paper_vqa.cli.paths import config_directory
from paper_vqa.evaluation.policies import (
    calibrate_policy_thresholds,
    compare_abstention_policies,
    flatten_policy_metrics,
    write_policy_comparison,
)
from paper_vqa.evaluation.reporting import read_generation_records
from paper_vqa.training.tracker import build_tracker


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Recalculate policy tables without loading a model or decoding images."""
    values = resolved_config(config)
    report_config = values["policy_report"]
    if str(report_config["split"]) != "validation":
        raise ValueError("saved-prediction policy recalibration is restricted to validation")
    predictions_path = report_config.get("predictions_path")
    if not predictions_path:
        raise ValueError("Set policy_report.predictions_path to a validation predictions.json")
    records = read_generation_records(Path(str(predictions_path)))
    evaluation_config = values["evaluation"]
    thresholds = calibrate_policy_thresholds(
        records,
        float(evaluation_config["minimum_answerable_recall"]),
        [float(target) for target in evaluation_config["matched_answerable_recall_targets"]],
    )
    comparison = compare_abstention_policies(
        records,
        thresholds,
        int(evaluation_config["risk_coverage_points"]),
    )
    output_directory = Path(str(report_config["output_dir"]))
    write_policy_comparison(comparison, output_directory)
    tracker = build_tracker(values["logging"], values)
    try:
        tracker.log(flatten_policy_metrics(comparison, "validation"), step=0)
        tracker.log_artifact(output_directory, "validation-policy-report", "evaluation")
    finally:
        tracker.finish()
    print(f"Policy report written to {output_directory}")


if __name__ == "__main__":
    main()
