import os
import random

import numpy as np
import pytest
import torch

from paper_vqa.utils.manifests import (
    ManifestEntry,
    assert_disjoint_manifests,
    manifest_from_entries,
)
from paper_vqa.utils.reproducibility import seed_everything


def test_manifest_overlap_is_rejected() -> None:
    train = manifest_from_entries("train", None, 42, [ManifestEntry("vizwiz", "train", "a")])
    test = manifest_from_entries("test", None, 42, [ManifestEntry("vizwiz", "test", "a")])

    with pytest.raises(ValueError, match="Data leakage"):
        assert_disjoint_manifests(train, test)


def test_seed_everything_reproduces_all_local_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    report = seed_everything(9)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    seed_everything(9)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert first == second
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert report.tokenizers_parallelism is False
    assert report.cpu_threads == 1
    assert torch.get_num_threads() == 1
