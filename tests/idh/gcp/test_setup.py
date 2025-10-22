from types import SimpleNamespace
from unittest.mock import MagicMock

import idh.gcp.setup as setup


def test_create_bigquery_dataset_success(monkeypatch):
    client = MagicMock()
    dataset = MagicMock(dataset_id="data", location="US")
    client.create_dataset.return_value = dataset

    monkeypatch.setattr(setup.bigquery, "Client", lambda project=None: client)
    monkeypatch.setattr(setup.bigquery, "Dataset", lambda full_id: dataset)

    assert setup.create_bigquery_dataset("proj", "dataset", "US") is True
    client.create_dataset.assert_called_once()


def test_create_bigquery_dataset_conflict(monkeypatch):
    client = MagicMock()
    client.create_dataset.side_effect = setup.Conflict("exists")
    dataset = MagicMock()

    monkeypatch.setattr(setup.bigquery, "Client", lambda project=None: client)
    monkeypatch.setattr(setup.bigquery, "Dataset", lambda full_id: dataset)

    assert setup.create_bigquery_dataset("proj", "dataset", "US") is True


def test_create_bigquery_dataset_error(monkeypatch):
    def fake_client(project=None):
        raise setup.GoogleCloudError("boom")

    monkeypatch.setattr(setup.bigquery, "Client", fake_client)

    assert setup.create_bigquery_dataset("proj", "dataset", "US") is False


def test_convert_csv_to_parquet(monkeypatch):
    calls = []

    class DummySchema:
        name = "dummy"

        def __call__(self):
            return self

        def to_parquet(self):
            calls.append("called")

    monkeypatch.setattr(setup, "SCHEMAS", [lambda: DummySchema()])

    setup.convert_csv_to_parquet()
    assert calls == ["called"]


def test_create_table_invokes_manager(monkeypatch):
    manager = MagicMock()

    class DummyTable:
        def __call__(self, **kwargs):
            return manager

    setup.create_table(MagicMock(), DummyTable(), "bucket", "prefix", "dataset")
    manager.create_table.assert_called_once()
    manager.load_from_gs.assert_called_once()


def test_create_data_tables_from_parquet(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(setup.bigquery, "Client", lambda: client)
    fake_config = SimpleNamespace(
        storage=SimpleNamespace(bucket="bucket", parquet_prefix="prefix"),
        dataset_name="dataset",
    )
    monkeypatch.setattr(setup, "config", fake_config)

    called = []

    def fake_create_table(client, table, bucket, prefix, dataset_id):
        called.append((client, table, bucket, prefix, dataset_id))

    monkeypatch.setattr(setup, "TABLES", [lambda **kwargs: MagicMock()])
    monkeypatch.setattr(setup, "create_table", fake_create_table)

    setup.create_data_tables_from_parquet()
    assert called


def test_load_csv_data_in_gcs(monkeypatch, tmp_path):
    download_dir = tmp_path / "temp"
    download_dir.mkdir()
    file_path = download_dir / "data.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(setup.fetch_data, "download_raw_csv_files", lambda: download_dir)
    uploaded = []
    monkeypatch.setattr(
        setup,
        "upload_to_gcs",
        lambda bucket, file_path, destination: uploaded.append((bucket, file_path, destination)),
    )
    monkeypatch.setattr(setup.fetch_data, "delete_temp_dir", lambda: uploaded.append("deleted"))

    setup.load_csv_data_in_gcs("bucket", "prefix")
    assert uploaded[0][2] == "prefix/data.csv"
    assert "deleted" in uploaded
