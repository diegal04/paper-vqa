"""Hydra entry point that materialises only manifests and validates no leakage."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.datamodule import VQADataModule
from paper_vqa.models.factory import build_processor


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Download/load configured partitions, validate them and save manifests."""
    values = resolved_config(config)
    processor = build_processor(str(values["model"]["name"]), values["model"].get("revision"))
    include_test = bool(values["data"].get("include_test", False))
    loaders = VQADataModule(values["data"], values["replay"], processor, values["trainer"]).build(
        include_test=include_test
    )
    output_directory = Path(str(values["trainer"]["output_dir"])).parent
    write_manifests(loaders.manifests, output_directory)
    print(f"Validated {len(loaders.manifests)} manifests in {output_directory}")


if __name__ == "__main__":
    main()
