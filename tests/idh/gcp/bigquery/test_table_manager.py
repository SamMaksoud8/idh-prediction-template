from unittest.mock import MagicMock

import idh.gcp.bigquery.table_manager as tm


def make_manager():
    client = MagicMock()
    manager = tm.BigQueryTableManager(
        client=client,
        project_id="proj",
        dataset_id="dataset",
        dataset_name="table",
        schema=["schema"],
        gs_bucket="bucket",
        gs_path="path/file",
    )
    return manager, client


def test_load_from_gs(monkeypatch):
    manager, client = make_manager()
    job = MagicMock()
    job.result.return_value = None
    client.load_table_from_uri.return_value = job
    client.get_table.return_value = MagicMock(num_rows=10)

    manager.load_from_gs()
    client.load_table_from_uri.assert_called_once()
    job.result.assert_called_once()
    client.get_table.assert_called_once_with("proj.dataset.table")


def test_create_table_handles_conflict(monkeypatch):
    manager, client = make_manager()
    client.create_table.side_effect = tm.Conflict("exists")

    manager.create_table()
    client.create_table.assert_called_once()


def test_subclasses_build_expected_schema():
    client = MagicMock()
    real = tm.RealTimeMachineDataTable(client)
    assert real.table_id.endswith("real-time-machine-data")

    demo = tm.PatientDemographicsTable(client)
    assert any(field.name == "pid" for field in demo.schema)

    reg = tm.RegistrationDataTable(client)
    assert any(field.name == "weightstart" for field in reg.schema)
