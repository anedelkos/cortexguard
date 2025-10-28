from pathlib import Path

import yaml

from kitchenwatch.common.constants import DEFAULT_SAMPLE_MANIFEST_PATH
from kitchenwatch.simulation.manifest_loader import ManifestLoader
from kitchenwatch.simulation.models.trial import Trial


def test_manifest_roundtrip(tmp_path: Path) -> None:
    """Ensure manifest loading and re-serialization is lossless."""

    # Load the sample manifest via ManifestLoader
    loader = ManifestLoader(DEFAULT_SAMPLE_MANIFEST_PATH)
    loader.load()

    # Sanity checks
    assert len(loader.trials) > 0, "Manifest must contain at least one trial"
    assert all(isinstance(t, Trial) for t in loader.trials)

    # Serialize back to YAML
    out_manifest_path = tmp_path / "roundtrip_manifest.yaml"
    with out_manifest_path.open("w") as f:
        yaml.safe_dump({"trials": [t.model_dump() for t in loader.trials]}, f)

    # Reload and compare
    reloaded_loader = ManifestLoader(out_manifest_path)
    reloaded_loader.load()

    assert len(reloaded_loader.trials) == len(loader.trials)
    for original, reloaded in zip(loader.trials, reloaded_loader.trials, strict=True):
        assert original == reloaded, "Round-trip manifest mismatch"
