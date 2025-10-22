from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import idh.gcp.storage.utils as storage_utils


def test_create_gcs_bucket_existing(monkeypatch):
    client = storage_utils.storage.Client(project="proj")
    client.create_bucket("existing")

    monkeypatch.setattr(storage_utils.storage, "Client", lambda project=None: client)

    assert storage_utils.create_gcs_bucket("existing", "proj", "US") is True


def test_create_gcs_bucket_new(monkeypatch):
    client = storage_utils.storage.Client(project="proj")
    monkeypatch.setattr(storage_utils.storage, "Client", lambda project=None: client)

    assert storage_utils.create_gcs_bucket("new-bucket", "proj", "US") is True
    assert "new-bucket" in client.buckets


def test_create_gcs_bucket_error(monkeypatch):
    def fake_client(project=None):
        raise storage_utils.GoogleCloudError("boom")

    monkeypatch.setattr(storage_utils.storage, "Client", fake_client)

    assert storage_utils.create_gcs_bucket("bad", "proj", "US") is False


def test_create_gspath():
    assert storage_utils.create_gspath("bucket", "prefix") == "gs://bucket/prefix"


def test_upload_to_gcs(monkeypatch, tmp_path):
    client = storage_utils.storage.Client()
    monkeypatch.setattr(storage_utils.storage, "Client", lambda: client)

    file_path = tmp_path / "file.txt"
    file_path.write_text("data", encoding="utf-8")

    storage_utils.upload_to_gcs("bucket", str(file_path), "dest.txt", timeout=10)

    blob = client.buckets["bucket"]._blobs["dest.txt"]
    assert blob.uploaded_from == (str(file_path), 10)
