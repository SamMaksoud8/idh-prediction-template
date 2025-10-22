from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import idh.gcp.bigquery.tables as tables


def set_fake_config(monkeypatch):
    fake_config = SimpleNamespace(
        project_name="proj",
        dataset_name="dataset",
        BigQuery=SimpleNamespace(
            features_dataset="features",
            sessionized_machine_data="sessionized",
            real_time_machine_data="real",
            registration_data="reg",
            patient_demographics="demo",
        ),
    )
    monkeypatch.setattr(tables, "config", fake_config)
    return fake_config


def test_table_id_helpers(monkeypatch):
    cfg = set_fake_config(monkeypatch)
    assert tables.features_table_id() == "proj.dataset.features"
    assert tables.sessionized_machine_data_table_id() == "proj.dataset.sessionized"
    assert tables.real_time_machine_data_table_id() == "proj.dataset.real"
    assert tables.registration_data_table_id() == "proj.dataset.reg"
    assert tables.patient_demographics_table_id() == "proj.dataset.demo"


def test_check_table_ready_success(monkeypatch):
    client = MagicMock()
    client.get_table.return_value = MagicMock()

    assert tables.check_table_ready(client, "proj.dataset.table", max_retries=1) is True
    client.get_table.assert_called_once()


def test_check_table_ready_not_found(monkeypatch):
    client = MagicMock()
    client.get_table.side_effect = tables.gcp_exceptions.NotFound("missing")
    monkeypatch.setattr(tables.time, "sleep", lambda s: None)

    assert tables.check_table_ready(client, "proj.dataset.table", max_retries=2) is False


def test_check_table_ready_other_error(monkeypatch):
    client = MagicMock()
    client.get_table.side_effect = RuntimeError("boom")
    monkeypatch.setattr(tables.time, "sleep", lambda s: None)

    assert tables.check_table_ready(client, "proj.dataset.table", max_retries=2) is False


def test_load_table_from_bigquery_success(monkeypatch):
    df = pd.DataFrame({"a": [1]})
    job = MagicMock()
    job.to_dataframe.return_value = df

    client = MagicMock()
    client.query.return_value = job

    result = tables.load_table_from_bigquery(client, "proj.dataset.table")
    assert result.equals(df)


def test_load_table_from_bigquery_failure(monkeypatch):
    client = MagicMock()
    client.query.side_effect = RuntimeError("boom")

    result = tables.load_table_from_bigquery(client, "proj.dataset.table")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_table_id(monkeypatch):
    set_fake_config(monkeypatch)
    assert tables.get_table_id("name") == "proj.dataset.name"


def test_load_specific_tables(monkeypatch):
    set_fake_config(monkeypatch)
    df = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(tables, "load_table_from_bigquery", lambda client, table_id: df)

    client = MagicMock()
    assert tables.load_real_time_machine_data(client).equals(df)
    assert tables.load_registration_data(client).equals(df)
    assert tables.load_patient_demographics(client).equals(df)
    assert tables.load_sessionized_machine_data(client).equals(df)
