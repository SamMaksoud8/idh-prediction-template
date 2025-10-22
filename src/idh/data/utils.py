"""Utility helpers for manipulating dialysis datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def convert_time_data_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise any known timestamp-like columns to ``datetime64`` objects."""
    df = df.copy()
    time_columns = [
        "datatime",
        "session_start_ts",
        "session_date",
        "keyindate",
        "dialysisstart",
        "dialysisend",
        "first_dialysis",
        "first_dialysis_ts",
        "keyindate_ts",
    ]
    active_time_columns = [i for i in df.columns if i in time_columns]
    for column in active_time_columns:
        df[column] = pd.to_datetime(df[column])
    return df


def dataframe_to_vertex_json(df: pd.DataFrame, feature_columns: Iterable[str]) -> dict[str, Any]:
    """Serialise ``feature_columns`` from ``df`` into the Vertex AI JSON schema."""
    prediction_df = df[list(feature_columns)].copy()

    instances_list = prediction_df.to_dict(orient="records")

    payload: dict[str, Any] = {"instances": instances_list}

    return payload


def load_machine_jsonl(jsonl_path: str | Path) -> pd.DataFrame:
    """Load machine telemetry from a JSON lines file and annotate session metadata."""
    data = load_jsonl_to_dataframe(jsonl_path)
    if data is None:
        return pd.DataFrame()
    data["datatime"] = pd.to_datetime(data["datatime"])
    session_starts = data.groupby("pid")["datatime"].transform("min")
    data["session_start_ts"] = session_starts
    data["session_date"] = data["session_start_ts"].dt.date
    return data


def load_jsonl_to_dataframe(file_path: str | Path) -> pd.DataFrame | None:
    """Load data from a JSON Lines file into a pandas DataFrame."""
    try:
        df = pd.read_json(file_path, lines=True)
        print(f"Successfully loaded {file_path} into a DataFrame.")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except ValueError as exc:
        print(f"Error decoding JSON from the file. Please check the file format. Details: {exc}")
        return None
    except Exception as exc:  # pragma: no cover - unexpected failure reporting
        print(f"An unexpected error occurred: {exc}")
        return None


def save_dict_to_json(payload_dict: dict[str, Any], file_path: str | Path) -> bool:
    """Persist ``payload_dict`` as a JSON document at ``file_path``."""
    try:
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(payload_dict, json_file, indent=2)

        print(f"Successfully saved payload to {file_path}")
        return True
    except IOError as exc:
        print(f"Error: Could not write to file at {file_path}. Details: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - unexpected failure reporting
        print(f"An unexpected error occurred: {exc}")
        return False
