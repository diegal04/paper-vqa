"""Hydra entry point for a single reproducible training run."""

from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import DictConfig

from paper_vqa.cli.common import resolve_device, resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.datamodule import VQADataModule
from paper_vqa.models.factory import build_processor, build_vqa_model
from paper_vqa.training.objective import MultitaskObjective, VQALossPolicy
from paper_vqa.training.tracker import build_tracker
from paper_vqa.training.trainer import Trainer, build_optimizer, build_scheduler
from paper_vqa.utils.reproducibility import seed_everything


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Train one configured VQA variant without evaluating the frozen test set."""
    values = resolved_config(config)
    trainer_config = values["trainer"]
    trainer_config["device"] = resolve_device(str(trainer_config["device"]))
    seed_everything(int(trainer_config["seed"]), bool(trainer_config["deterministic"]))
    processor = build_processor(str(values["model"]["name"]))
    datamodule = VQADataModule(values["data"], values["replay"], processor, trainer_config)
    loaders = datamodule.build()
    output_directory = Path(str(trainer_config["output_dir"])).parent
    write_manifests(loaders.manifests, output_directory)
    model = build_vqa_model(values["model"], values["head"])
    class_weights = values["loss"]["answerability_class_weights"]
    weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
    loss_policy = str(values["loss"]["vqa_loss_policy"])
    if loss_policy not in {"all_examples", "answerable_only"}:
        raise ValueError(f"unsupported VQA loss policy: {loss_policy}")
    objective = MultitaskObjective(
        answerability_weight=float(values["loss"]["answerability_weight"]),
        vqa_loss_policy=cast(VQALossPolicy, loss_policy),
        answerability_class_weights=weights,
    )
    optimizer = build_optimizer(model, trainer_config)
    scheduler = build_scheduler(
        optimizer,
        len(loaders.train) * int(trainer_config["epochs"]),
        trainer_config,
    )
    tracker = build_tracker(values["logging"], values)
    tracker.log_artifact(output_directory / "manifests", "data-manifests", "dataset")
    trainer = Trainer(model, objective, optimizer, scheduler, tracker, trainer_config, values)
    try:
        metrics = trainer.fit(loaders.train, loaders.validation)
        print(metrics)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
