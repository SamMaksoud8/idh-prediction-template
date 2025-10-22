from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import idh.gcp.bigquery.query as query


def test_run_query_wait(monkeypatch):
    client = MagicMock()
    job = MagicMock()
    client.query.return_value = job

    result = query.run_query(client, "SELECT 1")
    assert result is job
    job.result.assert_called_once()


def test_run_query_no_wait(monkeypatch):
    client = MagicMock()
    job = MagicMock()
    client.query.return_value = job

    result = query.run_query(client, "SELECT 1", wait=False)
    assert result is job
    job.result.assert_not_called()


def test_export_model_to_gcs(monkeypatch):
    captured = {}

    def fake_run_query(client, sql):
        captured["sql"] = sql
        return "job"

    monkeypatch.setattr(query, "run_query", fake_run_query)
    job = query.export_model_to_gcs(MagicMock(), "proj", "dataset", "model", "bucket")
    assert job == "job"
    assert "EXPORT MODEL `proj.dataset.model`" in captured["sql"]
    assert "gs://bucket/model-artifacts/" in captured["sql"]


def test_export_model_to_gcs_from_config(monkeypatch):
    fake_client = MagicMock()
    monkeypatch.setattr(query.bigquery, "Client", lambda project: fake_client)

    fake_config = SimpleNamespace(
        project_name="proj",
        dataset_name="dataset",
        model=SimpleNamespace(name="model"),
        storage=SimpleNamespace(bucket="bucket"),
    )
    monkeypatch.setattr(query, "config", fake_config)

    called = {}

    def fake_export(**kwargs):
        called.update(kwargs)
        return "job"

    monkeypatch.setattr(query, "export_model_to_gcs", lambda **kwargs: fake_export(**kwargs))

    job = query.export_model_to_gcs_from_config()
    assert job == "job"
    assert called["project_id"] == "proj"
    assert called["bucket"] == "bucket"


def test_features_engineering_calls_run_query(monkeypatch):
    captured = {}

    def fake_run_query(client, sql):
        captured["sql"] = sql
        return "job"

    monkeypatch.setattr(query, "run_query", fake_run_query)
    job = query.features_engineering(
        MagicMock(),
        "proj.dataset.features",
        "proj.dataset.reg",
        "proj.dataset.demo",
        "proj.dataset.session",
        15,
        3,
        5,
        90.0,
    )
    assert job == "job"
    assert "CREATE OR REPLACE TABLE `proj.dataset.features`" in captured["sql"]
    assert "proj.dataset.session" in captured["sql"]
