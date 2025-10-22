"""Helpers for loading, enriching, and serialising dialysis session data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from google.cloud import bigquery

import idh.data.preprocess as data_prepro
import idh.data.utils as data_utils
import idh.gcp.bigquery.query as bq_query
import idh.gcp.bigquery.tables as bq_tables
from idh.model import MODEL_FEATURES

SESSION_CSV_COLUMNS = [
    "pid",
    "datatime",
    "session_id",
    "first_dialysis_ts",
    "sbp",
    "dbp",
    "dia_temp_value",
    "conductivity",
    "uf",
    "blood_flow",
    "weightstart",
    "weightend",
    "dryweight",
    "temperature",
    "gender",
    "birthday",
    "DM",
]


def load_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV file containing session data and infer session start timestamps.

    Parameters
    ----------
    csv_path:
        Path to a CSV file using the :data:`SESSION_CSV_COLUMNS` schema.

    Returns
    -------
    pandas.DataFrame
        Session dataframe with ``session_start_ts`` computed for each row.
    """
    print(f"Loading session data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = data_utils.convert_time_data_to_datetime(df)
    df["session_start_ts"] = data_prepro.get_session_start_time(df)
    return df


def save_to_csv(df: pd.DataFrame, csv_path: str | Path) -> None:
    """Persist a session dataframe to disk with a stable column order."""
    print(f"Saving session data as {csv_path}...")
    df = df.copy()
    df = df[SESSION_CSV_COLUMNS]
    df.to_csv(csv_path, index=False)


def load_machine_data(client: bigquery.Client, session_id: str) -> pd.DataFrame:
    """Fetch machine telemetry for a specific session from BigQuery."""
    print(f"Fetching real time machine data for session_id={session_id}...")
    df = bq_query.get_session_machine_data(client, session_id)
    return df[df.session_id == session_id]


def load_registration_and_machine_data(client: bigquery.Client, session_id: str) -> pd.DataFrame:
    """Combine machine data with registration information for a session."""
    df_machine = load_machine_data(client, session_id)
    print("Loading patient registration data...")
    df_reg = bq_tables.load_registration_data(client)
    print("Preprocessing data...")
    return data_prepro.merge_machine_and_rego_data(df_machine, df_reg)


def load_session_data(client: bigquery.Client, session_id: str) -> pd.DataFrame:
    """Load and enrich all data required to score an individual session."""
    df_combined = load_registration_and_machine_data(client, session_id)
    print("Loading patient demographics data...")
    df_demo = bq_tables.load_patient_demographics(client)
    return data_prepro.merge_patient_demographics(df_combined, df_demo)


def session_data_to_vertex_json(df: pd.DataFrame) -> dict:
    """Convert a prepared session dataframe into the Vertex AI JSON payload format."""
    print("Creating aggregate features...")
    df_features = data_prepro.aggregate_features(df.copy())
    print("Saving data as vertex json...")
    return data_utils.dataframe_to_vertex_json(df_features, MODEL_FEATURES)


def csv_to_vertex_json(csv_path: str | Path) -> dict:
    """Shortcut that loads ``csv_path`` and returns the Vertex AI JSON payload."""
    df = load_from_csv(csv_path)
    return session_data_to_vertex_json(df)
