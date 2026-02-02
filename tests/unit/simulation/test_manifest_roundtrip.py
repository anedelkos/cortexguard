from pathlib import Path

import yaml

from cortexguard.common.constants import DEFAULT_SAMPLE_MANIFEST_PATH
from cortexguard.simulation.manifest_loader import ManifestLoader


def test_manifest_roundtrip(tmp_path: Path) -> None:
    """Ensure manifest loading and re-serialization is lossless."""

    # Load the sample manifest via ManifestLoader
    loader = ManifestLoader(DEFAULT_SAMPLE_MANIFEST_PATH)
    loader.load()

    # Sanity checks
    assert len(loader.trials) > 0, "Manifest must contain at least one trial"
    assert all(isinstance(t, type(loader.trials[0])) for t in loader.trials)

    # Convert Path objects → strings before YAML dump
    out_manifest_path = tmp_path / "roundtrip_manifest.yaml"
    with out_manifest_path.open("w") as f:
        yaml.safe_dump(
            {
                "trials": [
                    {k: str(v) if isinstance(v, Path) else v for k, v in t.model_dump().items()}
                    for t in loader.trials
                ]
            },
            f,
            sort_keys=False,
        )

    # Reload and compare
    reloaded_loader = ManifestLoader(out_manifest_path)
    reloaded_loader.load()

    assert len(reloaded_loader.trials) == len(loader.trials), "Trial count mismatch after roundtrip"
    for original, reloaded in zip(loader.trials, reloaded_loader.trials, strict=True):
        assert original == reloaded, f"Round-trip manifest mismatch for trial {original.trial_id}"
