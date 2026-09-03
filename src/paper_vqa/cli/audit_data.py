"""Hydra entry point for a descriptive, non-training VizWiz audit."""

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from paper_vqa.cli.common import resolved_config, write_manifests
from paper_vqa.cli.paths import config_directory
from paper_vqa.data.audit import DatasetAuditReport, VQADatasetAuditor
from paper_vqa.data.datasets import build_adapter
from paper_vqa.data.records import SourceConfig
from paper_vqa.utils.manifests import DatasetManifest, assert_disjoint_manifests


@hydra.main(version_base="1.3", config_path=config_directory(), config_name="config")
def main(config: DictConfig) -> None:
    """Audit configured VizWiz partitions and persist report plus manifests."""
    values = resolved_config(config)
    audit_config = values["audit"]
    sources = _sources(values["data"], bool(audit_config["include_test"]))
    examples_and_manifests = [_load_source(source) for source in sources]
    manifests = tuple(manifest for _, manifest in examples_and_manifests)
    assert_disjoint_manifests(*manifests)
    auditor = VQADatasetAuditor(
        preview_count=int(audit_config["preview_count"]),
        image_sample_count=int(audit_config["image_sample_count"]),
        top_answer_count=int(audit_config["top_answer_count"]),
    )
    seed = int(audit_config["seed"])
    report = DatasetAuditReport(
        seed=seed,
        partitions=tuple(
            auditor.audit(examples, seed + index)
            for index, (examples, _) in enumerate(examples_and_manifests)
        ),
    )
    output_directory = Path(str(values["trainer"]["output_dir"])).parent
    report.save(output_directory / str(audit_config["report_filename"]))
    write_manifests(manifests, output_directory)
    print(f"Audited {len(report.partitions)} partitions in {output_directory}")


def _sources(data_config: dict[str, Any], include_test: bool) -> tuple[dict[str, Any], ...]:
    """Return primary sources selected for a descriptive audit.

    Args:
        data_config: Resolved primary dataset configuration.
        include_test: Whether the frozen test split must be audited.

    Returns:
        Train and validation mappings, optionally followed by test.
    """
    source_names = ("train", "validation", "test") if include_test else ("train", "validation")
    return tuple(data_config[name] for name in source_names)


def _load_source(mapping: dict[str, Any]) -> tuple[list[Any], DatasetManifest]:
    """Load one source through its configured adapter.

    Args:
        mapping: Resolved source mapping.

    Returns:
        Examples and provenance manifest for the source.
    """
    return build_adapter(SourceConfig.from_mapping(mapping)).manifest()


if __name__ == "__main__":
    main()
