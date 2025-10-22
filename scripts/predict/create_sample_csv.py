"""
Create a CSV file containing a single session's data retrieved from BigQuery.
This script/module connects to BigQuery, loads the data for a specified session
identifier, and writes that session's data to a CSV file. The output CSV can be
used for local inspection or as input to model inference workflows (e.g.,
Vertex AI).
Behavior
- Initializes a BigQuery client using google.cloud.bigquery.Client.
- Loads session-level data via idh.data.session.load_session_data(client, session_id).
- Writes the resulting pandas.DataFrame to <save_dir>/<session_id>.csv using
    idh.data.session.save_to_csv.
Command-line arguments (also accepted by the create_sample_csv() function)
- --project-id: GCP project id used to initialize the BigQuery client.
    Default: "idh-prediction".
- --session-id: Session identifier to fetch and save (e.g., "1025914_383").
    Default: "1025914_383".
- --save-dir: Directory where the CSV will be written. Default: "data".
Examples
- As a module via CLI:
    python create_sample_csv.py --project-id my-project --session-id 12345_678 --save-dir ./data
- As a function:
    create_sample_csv(project_id="my-project", session_id="12345_678", save_dir="data")
Requirements and notes
- Expects the package-local modules under idh.* to be importable and configured.
- Requires credentials and permissions to access BigQuery (e.g., GOOGLE_APPLICATION_CREDENTIALS).
- If BigQuery retrieval fails or file writing fails, the script will raise the
    underlying exception (e.g., google.api_core.exceptions.GoogleAPICallError,
    IOError).
Side effects
- Creates (or overwrites) a CSV file at the path: <save_dir>/<session_id>.csv.
#1025914_118 idh
#1025914_383 no_idh
"""

import pandas as pd
from google.cloud import bigquery
import json
import idh.gcp.bigquery.tables as bq_tables
from idh.model import MODEL_FEATURES
import idh.data.utils as data_utils
import idh.data.preprocess as data_prepro
import idh.gcp.bigquery.query as bq_query
import argparse
import idh.data.session as session_data
from pathlib import Path


def create_sample_csv(project_id: str, session_id: str, save_dir: Path | str) -> Path:
    """
    Create a CSV file for a single session retrieved from BigQuery.

    Connects to BigQuery using the provided project_id, loads session data for
    `session_id`, and writes it to a CSV file at <save_dir>/<session_id>.csv.

    Args:
        project_id: GCP project id used to initialize the BigQuery client.
        session_id: Session identifier to fetch and save (e.g., "1025914_383").
        save_dir: Directory where the CSV will be written; may be a string or Path.

    Returns:
        Path: The full path to the written CSV file.

    Raises:
        google.api_core.exceptions.GoogleAPICallError: If BigQuery retrieval fails.
        OSError / IOError: If writing the CSV fails.
    """
    save_path = Path(save_dir) / f"{session_id}.csv"
    print("Initializing BigQuery client...")
    client = bigquery.Client(project=project_id)
    df = session_data.load_session_data(client, session_id)
    session_data.save_to_csv(df, save_path)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a vertex json for a given session id.")
    parser.add_argument(
        "--project-id",
        type=str,
        default="idh-prediction",
        help="The project id for the model.",
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default="1025914_383",
        help="The session id to convert to a vertex json.",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="data",
        help="The directory to save the vertex json.",
    )
    args = parser.parse_args()
    create_sample_csv(args.project_id, args.session_id, args.save_dir)
