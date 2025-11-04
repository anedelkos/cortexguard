import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError

from kitchenwatch.common.constants import DEFAULT_FULL_MANIFEST_PATH
from kitchenwatch.simulation.models.trial import Trial


class ManifestLoader:
    def __init__(
        self, path: Path = DEFAULT_FULL_MANIFEST_PATH, logger: logging.Logger | None = None
    ):
        """
        Manifest loader for simulation trials.

        Args:
            path: Path to the YAML manifest file.
            logger: Optional logger to use; defaults to module logger.
        """
        self.path = path
        self.logger = logger or logging.getLogger(__name__)
        self.trials: list[Trial] = []

    def load(self) -> list[Trial]:
        """Load and validate the manifest."""
        if not self.path.exists():
            raise FileNotFoundError(f"Manifest file not found at {self.path}")

        try:
            manifest_raw: dict[str, list[dict[str, object]]] = yaml.safe_load(self.path.read_text())
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML at {self.path}: {e}")
            raise ValueError(f"Invalid YAML format in manifest: {e}") from e

        if "trials" not in manifest_raw:
            raise ValueError("Manifest must contain a top-level 'trials' key")

        try:
            self.trials = TypeAdapter(list[Trial]).validate_python(manifest_raw["trials"])
        except ValidationError as e:
            self.logger.error(f"Validation failed for trials in manifest at {self.path}")

            for i, error in enumerate(e.errors()):
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                self.logger.debug(f"Trial #{i} failed at {loc}: {msg}")

            raise ValueError(f"Manifest contains invalid trial entries: {e}") from e

        return self.trials

    def get_trial_by_id(self, trial_id: str) -> Trial:
        """Retrieve a trial by its ID."""
        if not self.trials:
            raise ValueError("Manifest not loaded. Call `load()` first.")

        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial

        raise ValueError(f"Trial ID '{trial_id}' not found in manifest")

    def save(self, trials: list[Trial]) -> None:
        """Safely save updated trials to the manifest YAML file."""

        def _to_yaml_safe(value: Any) -> Any:
            if isinstance(value, (Path, HttpUrl)):
                return str(value)
            elif isinstance(value, BaseModel):
                return value.model_dump()
            elif isinstance(value, (list, tuple)):
                return [_to_yaml_safe(v) for v in value]
            elif isinstance(value, dict):
                return {k: _to_yaml_safe(v) for k, v in value.items()}

            return value

        data = {"trials": [_to_yaml_safe(t.model_dump()) for t in trials]}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            self.logger.info(f"Manifest saved successfully to {self.path}")

        except (OSError, yaml.YAMLError) as e:
            self.logger.error(f"Failed to save manifest to {self.path}: {e}")
            raise RuntimeError(f"Error writing manifest file: {self.path}") from e
