from pathlib import Path
from unittest.mock import MagicMock

import pytest

import idh.data.fetch as fetch


class DummyResponse:
    def __init__(self, content=b"data"):
        self.content = content
        self.iter_chunks = [content]

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for chunk in self.iter_chunks:
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummySession:
    def __init__(self, response=None):
        self.response = response or DummyResponse()
        self.called_with = None

    def get(self, url, stream=True):
        self.called_with = (url, stream)
        return self.response


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    target_dir = Path(fetch.__file__).resolve().parent / "temp"
    if target_dir.exists():
        fetch.delete_temp_dir(target_dir)
    monkeypatch.setattr(fetch, "Path", Path)
    yield target_dir
    if target_dir.exists():
        fetch.delete_temp_dir(target_dir)


def test_download_file_writes_chunks(tmp_path, monkeypatch):
    dummy_response = DummyResponse(content=b"chunk")

    def fake_get(url, stream=True, timeout=None):
        return dummy_response

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    destination = tmp_path / "file.bin"

    fetch.download_file("http://example.com/file", str(destination))
    assert destination.read_bytes() == b"chunk"


def test_download_raw_csv_files_creates_files(temp_dir, monkeypatch):
    created = []

    def fake_download(url, path):
        Path(path).write_text("csv", encoding="utf-8")
        created.append(path)

    monkeypatch.setattr(fetch, "download_file", fake_download)
    result_dir = fetch.download_raw_csv_files()

    assert result_dir == temp_dir
    for file_name in ["d1.csv", "idp.csv", "vip.csv"]:
        assert (temp_dir / file_name).exists()


def test_delete_temp_dir_success(temp_dir):
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "dummy.txt").write_text("data", encoding="utf-8")

    assert fetch.delete_temp_dir(temp_dir) is True
    assert not temp_dir.exists()


def test_delete_temp_dir_refuses_outside(tmp_path):
    outside_dir = tmp_path / "temp"
    outside_dir.mkdir()
    assert fetch.delete_temp_dir(outside_dir) is False
    assert outside_dir.exists()
