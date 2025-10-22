import json
from pathlib import Path

import pandas as pd
import pytest

from idh.data import utils as data_utils


def test_convert_time_data_to_datetime():
    df = pd.DataFrame(
        {
            "pid": [1, 2],
            "datatime": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            "session_start_ts": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
            "unrelated": [1, 2],
        }
    )

    converted = data_utils.convert_time_data_to_datetime(df)

    assert pd.api.types.is_datetime64_any_dtype(converted["datatime"])
    assert pd.api.types.is_datetime64_any_dtype(converted["session_start_ts"])
    assert not pd.api.types.is_datetime64_any_dtype(df["datatime"])


def test_dataframe_to_vertex_json():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    payload = data_utils.dataframe_to_vertex_json(df, ["a"])
    assert payload == {"instances": [{"a": 1}, {"a": 2}]}


def test_load_jsonl_to_dataframe_success(tmp_path):
    file_path = tmp_path / "data.jsonl"
    lines = [json.dumps({"a": 1}), json.dumps({"a": 2})]
    file_path.write_text("\n".join(lines), encoding="utf-8")

    df = data_utils.load_jsonl_to_dataframe(str(file_path))
    assert len(df) == 2
    assert df.iloc[0]["a"] == 1


def test_load_jsonl_to_dataframe_file_not_found():
    assert data_utils.load_jsonl_to_dataframe("missing.jsonl") is None


def test_load_jsonl_to_dataframe_invalid_json(tmp_path, capsys):
    file_path = tmp_path / "bad.jsonl"
    file_path.write_text("not json\n", encoding="utf-8")

    result = data_utils.load_jsonl_to_dataframe(str(file_path))
    captured = capsys.readouterr()

    assert result is None
    assert "Error decoding JSON" in captured.out


def test_load_machine_jsonl(tmp_path):
    file_path = tmp_path / "machine.jsonl"
    lines = [
        json.dumps({"pid": 1, "datatime": "2024-01-01T00:00:00"}),
        json.dumps({"pid": 1, "datatime": "2024-01-01T01:00:00"}),
    ]
    file_path.write_text("\n".join(lines), encoding="utf-8")

    df = data_utils.load_machine_jsonl(str(file_path))
    assert pd.api.types.is_datetime64_any_dtype(df["datatime"])
    assert "session_start_ts" in df
    assert (df["session_start_ts"].dt.hour == 0).all()
    assert "session_date" in df


def test_save_dict_to_json(tmp_path):
    payload = {"hello": "world"}
    file_path = tmp_path / "payload.json"

    success = data_utils.save_dict_to_json(payload, str(file_path))
    assert success is True
    assert json.loads(file_path.read_text(encoding="utf-8")) == payload
