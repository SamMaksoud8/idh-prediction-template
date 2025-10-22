import pandas as pd
import pytest

import idh.gcp.storage.data_schemas as schemas


def test_dataframe_schema_converts_types(monkeypatch):
    class DummySchema(schemas.DataFrameSchema):
        name: str = "dummy"
        schema: dict = {"value": "int64"}

    data = pd.DataFrame({"value": [1]})
    monkeypatch.setattr(DummySchema, "read_csv", lambda self: data)

    schema = DummySchema()
    assert schema.df["value"].dtype == "int64"


def test_patient_demographics_post_process(monkeypatch):
    data = pd.DataFrame(
        {
            "pid": [1],
            "gender": ["F"],
            "birthday": [1980],
            "first_dialysis": ["2020-01-01"],
            "DM": [1],
        }
    )
    monkeypatch.setattr(schemas.PatientDemographics, "read_csv", lambda self: data)

    schema = schemas.PatientDemographics()
    assert pd.api.types.is_datetime64_any_dtype(schema.df["first_dialysis"])
    assert schema.df["DM"].dtype == "bool"


def test_registration_data_post_process(monkeypatch):
    data = pd.DataFrame(
        {
            "pid": [1],
            "keyindate": ["2024-01-01"],
            "dialysisstart": ["08:00"],
            "dialysisend": ["24:30"],
            "weightstart": [70.0],
            "weightend": [69.0],
            "dryweight": [68.0],
            "temperature": [36.5],
        }
    )
    monkeypatch.setattr(schemas.RegistrationData, "read_csv", lambda self: data.copy())

    schema = schemas.RegistrationData()
    assert pd.api.types.is_datetime64_any_dtype(schema.df["dialysisstart"])
    assert pd.api.types.is_datetime64_any_dtype(schema.df["dialysisend"])
    assert schema.df["dialysisend"].dt.hour.iloc[0] == 0


def test_real_time_machine_data_post_process(monkeypatch):
    data = pd.DataFrame(
        {
            "pid": [1],
            "datatime": ["2024-01-01T00:00:00"],
            "measuretime": [1],
            "sbp": [100],
            "dbp": [70],
            "dia_temp_value": [36.5],
            "conductivity": [1.1],
            "uf": [0.5],
            "blood_flow": [300],
            "time": [1],
        }
    )
    monkeypatch.setattr(schemas.RealTimeMachineData, "read_csv", lambda self: data)

    schema = schemas.RealTimeMachineData()
    assert pd.api.types.is_datetime64_any_dtype(schema.df["datatime"])


def test_to_parquet_uses_dataframe(monkeypatch, tmp_path):
    class DummySchema(schemas.DataFrameSchema):
        name: str = "dummy"
        schema: dict = {}

    df = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(DummySchema, "read_csv", lambda self: df)
    called = {}

    def fake_to_parquet(path, index=False):
        called["path"] = path
        called["index"] = index

    monkeypatch.setattr(df, "to_parquet", fake_to_parquet)

    schema = DummySchema()
    schema.destination_path = "gs://bucket/parquet"
    schema.to_parquet()
    assert called["index"] is False
