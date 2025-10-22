from unittest.mock import MagicMock

import pandas as pd
import pytest

import idh.data.session as session


def make_session_df():
    return pd.DataFrame(
        {
            "pid": [1, 1],
            "datatime": ["2024-01-01 00:00:00", "2024-01-01 00:15:00"],
            "session_id": ["s1", "s1"],
            "first_dialysis": [1_600_000_000_000_000, 1_600_000_000_000_000],
            "first_dialysis_ts": pd.to_datetime(["2020-09-13 12:26:40", "2020-09-13 12:26:40"]),
            "sbp": [110, 105],
            "dbp": [70, 68],
            "dia_temp_value": [36.5, 36.7],
            "conductivity": [1.1, 1.2],
            "uf": [0.5, 0.6],
            "blood_flow": [300, 305],
            "weightstart": [70, 70],
            "weightend": [68, 68],
            "dryweight": [67, 67],
            "temperature": [36.6, 36.7],
            "gender": ["F", "F"],
            "birthday": [1980, 1980],
            "DM": [1, 1],
        }
    )


def test_load_from_csv(tmp_path):
    df = make_session_df()
    csv_path = tmp_path / "session.csv"
    df.to_csv(csv_path, index=False)

    loaded = session.load_from_csv(csv_path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["datatime"])
    assert "session_start_ts" in loaded.columns


def test_save_to_csv(tmp_path):
    df = make_session_df()
    csv_path = tmp_path / "saved.csv"
    session.save_to_csv(df, csv_path)

    saved = pd.read_csv(csv_path)
    assert list(saved.columns) == session.SESSION_CSV_COLUMNS


def test_load_machine_data(monkeypatch):
    fake_df = pd.DataFrame({"session_id": ["s1", "s2"], "value": [1, 2]})

    def fake_get(client, session_id):
        return fake_df

    monkeypatch.setattr(session.bq_query, "get_session_machine_data", fake_get)
    filtered = session.load_machine_data(MagicMock(), "s1")
    assert list(filtered["session_id"]) == ["s1"]


def test_load_registration_and_machine_data(monkeypatch):
    machine_df = pd.DataFrame(
        {
            "pid": [1],
            "session_id": ["s1"],
            "session_start_ts": pd.to_datetime(["2024-01-01 00:00"]),
        }
    )
    reg_df = pd.DataFrame(
        {
            "pid": [1],
            "keyindate": [1_704_067_200_000_000],
        }
    )

    monkeypatch.setattr(session, "load_machine_data", lambda client, sid: machine_df)
    monkeypatch.setattr(session.bq_tables, "load_registration_data", lambda client: reg_df)

    merged = session.load_registration_and_machine_data(MagicMock(), "s1")
    assert "session_date" in merged.columns


def test_load_session_data(monkeypatch):
    combined = pd.DataFrame(
        {
            "pid": [1],
            "session_start_ts": pd.to_datetime(["2024-01-01 00:00"]),
            "session_id": ["s1"],
        }
    )
    demo_df = pd.DataFrame({"pid": [1], "first_dialysis": [1_600_000_000_000_000]})

    monkeypatch.setattr(session, "load_registration_and_machine_data", lambda client, sid: combined)
    monkeypatch.setattr(session.bq_tables, "load_patient_demographics", lambda client: demo_df)

    merged = session.load_session_data(MagicMock(), "s1")
    assert "first_dialysis_ts" in merged.columns


def test_session_data_to_vertex_json(monkeypatch):
    features_df = pd.DataFrame(
        {feature: [idx] for idx, feature in enumerate(session.MODEL_FEATURES)}
    )
    monkeypatch.setattr(session.data_prepro, "aggregate_features", lambda df: features_df)
    monkeypatch.setattr(
        session.data_utils,
        "dataframe_to_vertex_json",
        lambda df, cols: {"instances": df[cols].to_dict("records")},
    )

    payload = session.session_data_to_vertex_json(make_session_df())
    assert set(payload["instances"][0].keys()) == set(session.MODEL_FEATURES)


def test_csv_to_vertex_json(monkeypatch, tmp_path):
    df = make_session_df()
    csv_path = tmp_path / "session.csv"
    df.to_csv(csv_path, index=False)

    called = {}

    def fake_session_data_to_vertex_json(df):
        called["called"] = True
        return {"instances": []}

    monkeypatch.setattr(session, "session_data_to_vertex_json", fake_session_data_to_vertex_json)

    payload = session.csv_to_vertex_json(csv_path)
    assert called.get("called") is True
    assert payload == {"instances": []}
