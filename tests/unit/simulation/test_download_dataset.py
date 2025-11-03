import io
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
from data.download_dataset import DatasetDownloader


def make_downloader(tmp_path: Path) -> DatasetDownloader:
    """Create a downloader with a temporary manifest."""
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("trials: []")
    return DatasetDownloader(manifest_path=manifest_path, data_dir=tmp_path)


def test_download_and_extract_success(tmp_path: Path) -> None:
    downloader = make_downloader(tmp_path)
    url = "http://example.com/sample.tar.gz"
    target_dir = tmp_path / "target"
    mock_tar = MagicMock()

    fake_response = io.BytesIO(b"fake content")

    with (
        patch(
            "data.download_dataset.urlopen",
            return_value=MagicMock(__enter__=lambda s: fake_response, __exit__=lambda *a: None),
        ) as mock_urlopen,
        patch("tarfile.open", return_value=mock_tar) as mock_open,
        patch("builtins.open", create=True),
    ):
        downloader._download_and_extract(url, target_dir)

    mock_urlopen.assert_called_once()
    mock_open.assert_called_once_with(target_dir / "sample.tar.gz", "r:gz")
    mock_tar.__enter__().extractall.assert_called_once_with(path=target_dir)


def test_download_and_extract_skips_if_dir_exists(tmp_path: Path) -> None:
    downloader = make_downloader(tmp_path)
    url = "http://example.com/skip.tar.gz"
    target_dir = tmp_path / "target"
    (target_dir / "existing_folder").mkdir(parents=True)

    with (
        patch("data.download_dataset.urlretrieve") as mock_url,
        patch("tarfile.open") as mock_open,
    ):
        downloader._download_and_extract(url, target_dir)

    mock_url.assert_not_called()
    mock_open.assert_not_called()


def test_download_and_extract_http_error(tmp_path: Path) -> None:
    downloader = make_downloader(tmp_path)
    url = "http://example.com/error.tar.gz"
    target_dir = tmp_path / "target"

    with patch("data.download_dataset.urlretrieve") as mock_url:
        mock_url.side_effect = URLError("Network fail")
        with pytest.raises(RuntimeError):
            downloader._download_and_extract(url, target_dir)
