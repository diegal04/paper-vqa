"""Object-oriented training loop for reproducible multitask VQA experiments."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from paper_vqa.models.selective_vqa import SelectiveVQAModel
from paper_vqa.training.checkpoints import CheckpointManager, CheckpointMetadata
from paper_vqa.training.objective import MultitaskObjective
from paper_vqa.training.tracker import Tracker

MonitorMode = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Scalar results emitted after one train or validation epoch."""

    loss: float
    generation_loss: float
    answerability_loss: float
    answerability_ap: float

    def prefixed(self, prefix: str) -> dict[str, float]:
        """Return metrics under stable W&B and checkpoint monitor names."""
        return {
            f"{prefix}/loss": self.loss,
            f"{prefix}/generation_loss": self.generation_loss,
            f"{prefix}/answerability_loss": self.answerability_loss,
            f"{prefix}/answerability_ap": self.answerability_ap,
        }


class Trainer:
    """Train a model, select checkpoints on validation only, and log every epoch."""

    def __init__(
        self,
        model: SelectiveVQAModel,
        objective: MultitaskObjective,
        optimizer: Optimizer,
        scheduler: Any | None,
        tracker: Tracker,
        config: Mapping[str, Any],
        resolved_config: dict[str, Any],
    ) -> None:
        """Configure the training loop and safe checkpoint monitor.

        Args:
            model: Model variant to optimise.
            objective: Configured multitask loss.
            optimizer: Optimizer over all intended trainable parameters.
            scheduler: Optional step-wise learning-rate scheduler.
            tracker: Event logger, W&B or no-op.
            config: Trainer settings from Hydra.
            resolved_config: Full resolved configuration saved in checkpoint metadata.
        """
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tracker = tracker
        self.config = config
        self.resolved_config = resolved_config
        self.device = torch.device(config["device"])
        self.model.to(self.device)
        self.checkpoints = CheckpointManager(Path(str(config["output_dir"])))
        self.monitor = str(config["checkpoint_monitor"])
        mode = str(config["checkpoint_mode"])
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint_mode must be 'min' or 'max'")
        self.monitor_mode = cast(MonitorMode, mode)
        self.best_value = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.epochs_without_improvement = 0

    def fit(
        self,
        train_loader: DataLoader[dict[str, Tensor]],
        validation_loader: DataLoader[dict[str, Tensor]],
    ) -> dict[str, float]:
        """Optimise for configured epochs and keep only validation-selected weights."""
        global_step = 0
        best_metrics: dict[str, float] = {}
        for epoch in range(1, int(self.config["epochs"]) + 1):
            train_metrics, global_step = self._run_epoch(
                train_loader, training=True, global_step=global_step, epoch=epoch
            )
            validation_metrics, global_step = self._run_epoch(
                validation_loader, training=False, global_step=global_step, epoch=epoch
            )
            combined = train_metrics.prefixed("train") | validation_metrics.prefixed("val")
            combined["epoch"] = float(epoch)
            self.tracker.log(combined, global_step)
            monitored = combined.get(self.monitor)
            if monitored is None or np.isnan(monitored):
                raise ValueError(f"checkpoint monitor {self.monitor!r} is unavailable")
            if self._improved(monitored):
                self.best_value = monitored
                self.epochs_without_improvement = 0
                self.checkpoints.save(
                    self.model,
                    CheckpointMetadata(
                        epoch=epoch,
                        monitor_name=self.monitor,
                        monitor_value=monitored,
                        threshold=None,
                        config=self.resolved_config,
                    ),
                )
                self.tracker.log_artifact(
                    self.checkpoints.directory,
                    name="selected-checkpoint",
                    artifact_type="model",
                )
                best_metrics = combined
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= int(self.config["early_stopping_patience"]):
                    print(self._epoch_summary(epoch, train_metrics, validation_metrics, monitored))
                    break
            print(self._epoch_summary(epoch, train_metrics, validation_metrics, monitored))
        return best_metrics

    def _run_epoch(
        self,
        loader: DataLoader[dict[str, Tensor]],
        training: bool,
        global_step: int,
        epoch: int,
    ) -> tuple[EpochMetrics, int]:
        """Run one mode-specific epoch and collect answerability AP without thresholds."""
        self.model.train(training)
        totals = {"loss": 0.0, "generation": 0.0, "answerability": 0.0, "examples": 0}
        answerability_labels: list[int] = []
        answerability_scores: list[float] = []
        progress = self._progress(loader, training, epoch)
        update_every = int(self.config.get("progress", {}).get("update_every_n_steps", 1))
        if update_every < 1:
            raise ValueError("progress.update_every_n_steps must be at least one")
        for batch_index, batch in enumerate(progress, start=1):
            moved = {
                name: value.to(self.device, non_blocking=True) for name, value in batch.items()
            }
            with torch.set_grad_enabled(training):
                output = self.model(
                    pixel_values=moved["pixel_values"],
                    input_ids=moved["input_ids"],
                    attention_mask=moved["attention_mask"],
                    labels=moved["labels"],
                )
                losses = self.objective(output, moved)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    losses.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), float(self.config["max_grad_norm"])
                    )
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
            batch_size = int(moved["pixel_values"].shape[0])
            totals["loss"] += float(losses.total.detach()) * batch_size
            totals["generation"] += float(losses.generation.detach()) * batch_size
            totals["answerability"] += float(losses.answerability.detach()) * batch_size
            totals["examples"] += batch_size
            if output.answerability_probabilities is not None:
                mask = moved["has_answerable"].bool()
                answerability_labels.extend(moved["answerable"][mask].detach().cpu().tolist())
                answerability_scores.extend(
                    output.answerability_probabilities[mask, 1].detach().cpu().tolist()
                )
            if training:
                global_step += 1
            if batch_index % update_every == 0:
                progress.set_postfix(
                    loss=f"{totals['loss'] / totals['examples']:.4f}",
                    generation=f"{totals['generation'] / totals['examples']:.4f}",
                )
        count = max(totals["examples"], 1)
        ap = (
            float(average_precision_score(answerability_labels, answerability_scores))
            if len(set(answerability_labels)) == 2
            else float("nan")
        )
        return (
            EpochMetrics(
                loss=totals["loss"] / count,
                generation_loss=totals["generation"] / count,
                answerability_loss=totals["answerability"] / count,
                answerability_ap=ap,
            ),
            global_step,
        )

    def _improved(self, value: float) -> bool:
        """Compare a monitor value with the selected direction."""
        return value < self.best_value if self.monitor_mode == "min" else value > self.best_value

    def _progress(self, loader: DataLoader[dict[str, Tensor]], training: bool, epoch: int) -> Any:
        """Create the configured terminal progress bar for one epoch.

        Args:
            loader: Batches in the current phase.
            training: Whether gradients are enabled for this phase.
            epoch: One-indexed epoch number shown to the user.

        Returns:
            A tqdm iterator that can be disabled through YAML.
        """
        settings = self.config.get("progress", {})
        phase = "Train" if training else "Validation"
        return tqdm(
            loader,
            desc=f"{phase} {epoch}/{self.config['epochs']}",
            total=len(loader),
            disable=not bool(settings.get("enabled", False)),
            leave=bool(settings.get("leave", False)),
        )

    def _epoch_summary(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        validation_metrics: EpochMetrics,
        monitored: float,
    ) -> str:
        """Build one readable console line after train and validation complete.

        Args:
            epoch: One-indexed completed epoch.
            train_metrics: Aggregated training metrics.
            validation_metrics: Aggregated validation metrics.
            monitored: Value used for checkpoint selection.

        Returns:
            Human-readable epoch summary for the terminal.
        """
        return (
            f"Epoch {epoch}/{self.config['epochs']} | "
            f"train_loss={train_metrics.loss:.4f} | val_loss={validation_metrics.loss:.4f} | "
            f"{self.monitor}={monitored:.4f}"
        )


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> Optimizer:
    """Build AdamW over explicitly trainable parameters only."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model exposes no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def build_scheduler(optimizer: Optimizer, total_steps: int, config: Mapping[str, Any]) -> Any:
    """Build the configured linear warmup/decay scheduler."""
    from transformers import get_scheduler

    warmup_steps = int(config["warmup_steps"])
    return get_scheduler(
        name=str(config["scheduler"]),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
