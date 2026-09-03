"""Locate packaged Hydra configuration in editable and wheel installations."""

from importlib.resources import files
from pathlib import Path


def config_directory() -> str:
    """Return the installed config directory, falling back to the source tree."""
    packaged = files("paper_vqa").joinpath("configs")
    if packaged.is_dir():
        return str(packaged)
    return str(Path(__file__).resolve().parents[3] / "configs")
