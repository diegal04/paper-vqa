"""Deterministic execution helpers."""

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

_THREADPOOL_LIMITER: Any | None = None


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    """Record of deterministic settings applied to one run."""

    seed: int
    deterministic_algorithms: bool
    cuda_available: bool
    tokenizers_parallelism: bool
    cpu_threads: int


def seed_everything(
    seed: int, deterministic: bool = True, cpu_threads: int = 1
) -> ReproducibilityReport:
    """Seed all local random number generators used by this project.

    Args:
        seed: Non-negative seed shared by Python, NumPy and PyTorch.
        deterministic: Whether to request deterministic PyTorch algorithms.
        cpu_threads: Maximum threads used by CPU numerical backends.

    Returns:
        A serialisable record of the applied settings.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    thread_count = str(cpu_threads)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = thread_count
    global _THREADPOOL_LIMITER
    _THREADPOOL_LIMITER = threadpool_limits(limits=cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(cpu_threads)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work starts.
        # Repeated seed calls in one test process retain the first configured value.
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return ReproducibilityReport(seed, deterministic, torch.cuda.is_available(), False, cpu_threads)


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
