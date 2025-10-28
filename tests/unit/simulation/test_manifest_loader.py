from pathlib import Path

import pytest
from _pytest.logging import LogCaptureFixture

from kitchenwatch.simulation.manifest_loader import ManifestLoader
from kitchenwatch.simulation.models.trial import Trial

VALID_MANIFEST = """
trials:
  - trial_id: "1"
    subject: "subject_1"
    food_type: "bagel"
    trial_num: 1
    sensor_file: "data/sample/subject_1_bagel/wrenches_poses_transforms/1_wrenches_poses.csv"
    image_folder: "data/sample/subject_1_bagel/rgb_images/1/"
    depth_folder: "data/sample/subject_1_bagel/depth_images/1/"
    fusion_window: 0.03
    anomaly_scenario: "none"
    seed: 42
"""

INVALID_MANIFESTS = [
    # malformed YAML
    ("not: [valid: yaml", ValueError),
    # missing 'trials' key
    ("not_trails_key: []", ValueError),
    # invalid trial entry (missing required field 'trial_id')
    (
        """
        trials:
          - subject: "subject_1"
            food_type: "bagel"
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
    f = tmp_path / "invalid.yaml"
    f.write_text(manifest_content)
    loader = ManifestLoader(f)
    with caplog.at_level("DEBUG"):
        with pytest.raises(expected_exception):
            loader.load()

        # If it was a trial-level validation error, ensure debug log exists
        if "trials" in manifest_content or "subject" in manifest_content:
            assert (
                "failed" in caplog.text
                or "Manifest must contain a top-level 'trials'" in caplog.text
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
