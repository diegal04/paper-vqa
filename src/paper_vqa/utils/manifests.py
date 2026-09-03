"""Dataset manifests used to detect data leakage and retain provenance."""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """A source-specific sample identifier included in an experiment split."""

    dataset: str
    split: str
    sample_id: str

    @property
    def key(self) -> tuple[str, str]:
        """Return the identifier used for cross-split leakage detection."""
        return self.dataset, self.sample_id


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable record of samples and source revision used by a run."""

    name: str
    revision: str | None
    seed: int
    entries: tuple[ManifestEntry, ...]

    def save(self, path: Path) -> None:
        """Write the manifest as deterministic JSON.

        Args:
            path: Target JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["entries"] = [asdict(entry) for entry in self.entries]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def assert_disjoint_manifests(*manifests: DatasetManifest) -> None:
    """Raise when a sample appears in more than one supplied manifest.

    Args:
        manifests: Train, validation, test, or replay manifests to compare.

    Raises:
        ValueError: If a sample ID from the same source appears twice.
    """
    seen: dict[tuple[str, str], str] = {}
    for manifest in manifests:
        for entry in manifest.entries:
            previous = seen.get(entry.key)
            if previous is not None:
                raise ValueError(
                    f"Data leakage: {entry.dataset}:{entry.sample_id} appears in "
                    f"both {previous!r} and {manifest.name!r}."
                )
            seen[entry.key] = manifest.name


def manifest_from_entries(
    name: str,
    revision: str | None,
    seed: int,
    entries: Iterable[ManifestEntry],
) -> DatasetManifest:
    """Create a manifest while preserving the supplied entry order."""
    return DatasetManifest(name, revision, seed, tuple(entries))
