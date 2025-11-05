"""
Dataset downloader for KitchenWatch simulation.

Downloads and extracts dataset archives defined in the simulation manifest.
Each trial entry may include a `download_files` block with URLs and target directories.

Usage:
    python -m kitchenwatch.simulation.download_dataset --manifest demo/full_dataset_manifest.yaml
"""

import argparse
import logging
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kitchenwatch.common.constants import (
    DEFAULT_FULL_MANIFEST_PATH,
    DEFAULT_RAW_DATA_PATH,
)
from kitchenwatch.simulation.manifest_loader import ManifestLoader


def setup_logger() -> logging.Logger:
    """Configure a simple console logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("dataset_downloader")


class DatasetDownloader:
    """Handles dataset download and extraction from a YAML manifest."""

    def __init__(
        self,
        manifest_path: Path,
        data_dir: Path = DEFAULT_RAW_DATA_PATH,
        logger: logging.Logger | None = None,
    ):
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger(__name__)

    def run(self) -> None:
        """Execute the full download + extract process."""
        loader = ManifestLoader(self.manifest_path, logger=self.logger)
        trials = loader.load()

        if not trials:
            self.logger.warning("No trials found in manifest.")
            return

        for trial in trials:
            trial_id = getattr(trial, "trial_id", "<unknown>")
            downloads = getattr(trial, "download_files", None)

            if not downloads:
                self.logger.info(f"Skipping {trial_id} — no download_files defined.")
                continue

            self.logger.info(f"Processing trial {trial_id}...")

            for name, entry in downloads.items():
                url = str(getattr(entry, "url", None) or "")
                target_dir = Path(getattr(entry, "target_dir", self.data_dir) or self.data_dir)

                if not url:
                    self.logger.warning(f"Missing URL for {trial_id}/{name}, skipping.")
                    continue

                try:
                    self._download_and_extract(url, target_dir)
                except Exception as e:
                    self.logger.error(f"Failed to process {trial_id}/{name}: {e}")
                    raise e

    def _download_and_extract(self, url: str, target_dir: Path) -> None:
        """Download and extract a .tar.gz archive. Skip if files already exist."""
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            url.split("filename%2A%3DUTF-8%27%27")[-1].split("&")[0]
            if "filename%2A%3DUTF-8%27%27" in url
            else url.split("/")[-1]
        )
        tmp_path = target_dir / filename

        # Check if target_dir already contains content (trial folder exists)
        existing_subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
        if existing_subdirs:
            self.logger.info(
                f"✓ Skipping {filename} — target folders already exist in {target_dir}"
            )
            return

        self.logger.info(f"→ Downloading {filename} to {tmp_path}")

        try:
            # --- Add custom User-Agent to avoid Dataverse 403 ---
            req = Request(url, headers={"User-Agent": "Wget/1.21.4"})
            with urlopen(req) as response, open(tmp_path, "wb") as out_file:
                out_file.write(response.read())
        except HTTPError as e:
            raise RuntimeError(f"HTTP error {e.code} while downloading {url}") from e
        except URLError as e:
            raise RuntimeError(f"Failed to reach server for {url}: {e.reason}") from e

        self.logger.info(f"✓ Download complete — extracting to {target_dir}")

        try:
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
        except tarfile.TarError as e:
            raise RuntimeError(f"Error extracting archive {tmp_path}: {e}") from e
        finally:
            # Optional: delete the tar.gz after extraction
            try:
                tmp_path.unlink()
            except OSError:
                self.logger.warning(f"Could not remove temporary file {tmp_path}")

        self.logger.info(f"✓ Finished {filename} → {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract datasets from https://robotfeeding.io/datasets."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_FULL_MANIFEST_PATH,
        help="Path to dataset manifest (YAML).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Base directory for data (default: current).",
    )
    args = parser.parse_args()

    logger = setup_logger()
    downloader = DatasetDownloader(
        manifest_path=args.manifest, data_dir=args.data_dir, logger=logger
    )

    try:
        downloader.run()
        logger.info("✅ All datasets processed successfully.")
    except Exception as e:
        logger.error(f"❌ Download process failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
