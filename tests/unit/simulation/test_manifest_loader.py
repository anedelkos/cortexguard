from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

from kitchenwatch.simulation.manifest_loader import ManifestLoader
from kitchenwatch.simulation.models.trial import Trial

VALID_MANIFEST = """
trials:
  - trial_id: "1"
    rgb_path: "rgb/1.png"
    depth_path: null
    timestamp_ns: 123
    force_x: 1.0
    force_y: 0.0
    force_z: 0.0
    torque_x: 0.1
    torque_y: 0.2
    torque_z: 0.3
    pos_x: 0.0
    pos_y: 0.0
    pos_z: 0.0
"""

INVALID_MANIFESTS = [
    # malformed YAML
    ("not: [valid: yaml", ValueError),
    # missing 'trials' key
    ("not_trails_key: []", ValueError),
    # invalid trial field type
    (
        """
        trials:
        - trial_id: "1"
          rgb_path: "rgb/1.png"
          depth_path: null
          timestamp_ns: "not-an-int"
          force_x: 1.0
          force_y: 0.0
          force_z: 0.0
          torque_x: 0.1
          torque_y: 0.2
          torque_z: 0.3
          pos_x: 0.0
          pos_y: 0.0
          pos_z: 0.0
        """,
        ValueError,
    ),
]


@pytest.fixture
def valid_manifest_file(tmp_path: Path) -> Path:
    f = tmp_path / "valid_manifest.yaml"
    f.write_text(VALID_MANIFEST)
    return f


@pytest.mark.parametrize("manifest_content,expected_exception", INVALID_MANIFESTS)
def test_load_invalid_manifests(
    tmp_path: Path, manifest_content: str, expected_exception: type, caplog: LogCaptureFixture
) -> None:
    """Parametrized test for all invalid manifests including logging."""
    f = tmp_path / "invalid.yaml"
    f.write_text(manifest_content)
    loader = ManifestLoader(f)
    with caplog.at_level("DEBUG"):
        with pytest.raises(expected_exception):
            loader.load()

    # For validation errors, assert debug logs exist
    if (
        expected_exception is ValueError
        and "trials" in manifest_content
        or "not-an-int" in manifest_content
    ):
        assert (
            "Trial #0 failed at" in caplog.text
            or "Manifest must contain a top-level 'trials' key" in caplog.text
        )


def test_load_valid_manifest(valid_manifest_file: Path) -> None:
    loader = ManifestLoader(valid_manifest_file)
    trials = loader.load()
    assert len(trials) == 1
    trial = trials[0]
    assert isinstance(trial, Trial)
    assert trial.trial_id == "1"


def test_get_trial_by_id_success(valid_manifest_file: Path) -> None:
    loader = ManifestLoader(valid_manifest_file)
    loader.load()
    trial = loader.get_trial_by_id("1")
    assert trial.trial_id == "1"


def test_get_trial_by_id_not_found(valid_manifest_file: Path) -> None:
    loader = ManifestLoader(valid_manifest_file)
    loader.load()
    with pytest.raises(ValueError):
        loader.get_trial_by_id("nonexistent_id")


def test_get_trial_by_id_without_load() -> None:
    loader = ManifestLoader(Path("/some/path.yaml"))
    with pytest.raises(ValueError):
        loader.get_trial_by_id("1")


def test_missing_file_raises() -> None:
    loader = ManifestLoader(Path("/non/existent/file.yaml"))
    with pytest.raises(FileNotFoundError):
        loader.load()
