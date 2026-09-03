"""Hydra entry point for validation calibration and one frozen-test evaluation."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolve_device, resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.datamodule import VQADataModule
from paper_vqa.evaluation.metrics import select_safety_threshold
from paper_vqa.evaluation.runner import Evaluator, write_evaluation
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
    seed_everything(int(trainer_config["seed"]), bool(trainer_config["deterministic"]))
    processor = build_processor(str(values["model"]["name"]), values["model"].get("revision"))
    loaders = VQADataModule(values["data"], values["replay"], processor, trainer_config).build(
        include_test=True
    )
    if not loaders.test_examples:
        raise ValueError("frozen test evaluation requires a non-empty test partition")
    output_directory = Path(str(trainer_config["output_dir"])).parent
    write_manifests(loaders.manifests, output_directory)
    model = build_vqa_model(values["model"], values["head"])
    CheckpointManager(Path(str(checkpoint_path))).load(model)
    evaluator = Evaluator(model, processor, str(trainer_config["device"]))
    configured_threshold = values["evaluation"]["threshold"]
    if configured_threshold is not None:
        threshold = float(configured_threshold)
    elif model.answerability_head is None:
        threshold = 0.0
    else:
        labels, scores = evaluator.calibration_scores(loaders.validation)
        threshold = select_safety_threshold(
            labels, scores, float(values["evaluation"]["minimum_answerable_recall"])
        ).threshold
    result, predictions = evaluator.evaluate_examples(
        loaders.test_examples,
        threshold,
        dict(values["evaluation"]["generation"]),
        int(values["evaluation"]["ece_bins"]),
    )
    write_evaluation(result, predictions, output_directory / "evaluation")
    tracker = build_tracker(values["logging"], values)
    try:
        tracker.log_artifact(output_directory / "manifests", "data-manifests", "dataset")
        tracker.log_artifact(output_directory / "evaluation", "frozen-test-results", "evaluation")
        tracker.log({"test/vqa_accuracy": result.vqa_accuracy}, step=0)
    finally:
        tracker.finish()
    print(result)


if __name__ == "__main__":
    main()
