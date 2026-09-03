"""Deterministic execution helpers."""

import os
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    """Record of deterministic settings applied to one run."""

    seed: int
    deterministic_algorithms: bool
    cuda_available: bool


def seed_everything(seed: int, deterministic: bool = True) -> ReproducibilityReport:
    """Seed all local random number generators used by this project.

    Args:
        seed: Non-negative seed shared by Python, NumPy and PyTorch.
        deterministic: Whether to request deterministic PyTorch algorithms.

    Returns:
        A serialisable record of the applied settings.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return ReproducibilityReport(seed, deterministic, torch.cuda.is_available())


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """Create a DataLoader worker seeding callback derived from ``seed``.

    Args:
        seed: Base experiment seed.

    Returns:
        A callback suitable for ``DataLoader(worker_init_fn=...)``.
    """

    def initialise_worker(worker_id: int) -> None:
        """Seed one worker without sharing RNG state with other workers."""
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return initialise_worker


def make_generator(seed: int) -> torch.Generator:
    """Create a seeded PyTorch generator for deterministic sampling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
