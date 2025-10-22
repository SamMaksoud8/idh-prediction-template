"""
create_vertex_json.py
Module for creating a Vertex AI-compatible JSON payload for a single patient
session. The module can load session data either from a local CSV file or from
BigQuery, convert the session DataFrame into a Vertex JSON payload, and save
that payload to disk for use in inference testing.
Functions
---------
create_vertex_json(project_id, session_id, csv_path, save_path)
    Build and persist a Vertex JSON payload for the provided session.
    - If csv_path is provided, session data is loaded from the CSV via
      idh.data.session.load_from_csv.
    - Otherwise, a BigQuery client is initialised (via google.cloud.bigquery.Client)
      and session data is loaded from BigQuery via idh.data.session.load_session_data.
    - The session DataFrame is converted to a Vertex payload with
      idh.data.session.session_data_to_vertex_json and saved using
      idh.data.utils.save_dict_to_json.
Parameters
----------
project_id : str
    Google Cloud project id used to initialise the BigQuery client when
    loading session data from BigQuery. Typical default: "idh-prediction".
session_id : str
    Session identifier to convert to a Vertex JSON payload (e.g. "1025914_383").
csv_path : str or None
    Optional path to a CSV file containing the session data. If provided,
    data is loaded from this CSV instead of BigQuery. If None, BigQuery is used.
save_path : str
    Path where the generated Vertex JSON payload will be written. If the file
    exists it will be overwritten.
Returns
-------
None
    The function writes the payload to disk at save_path. No value is returned.
Side effects
------------
- May initialise a BigQuery client and perform queries if csv_path is not given.
- Writes a JSON file to disk.
- Relies on helper modules:
    idh.data.session, idh.data.utils
Usage (CLI)
-----------
The module is intended to be executable as a script. Typical invocation:
python create_vertex_json.py --project-id idh-prediction --session-id 1025914_383 --save-path payload.json
To load from a local CSV instead of BigQuery:
python create_vertex_json.py --csv /path/to/session.csv --save-path payload.json
Notes
-----
- The exact JSON structure produced depends on idh.data.session.session_data_to_vertex_json
  and must conform to the Vertex AI prediction request format expected by the target model.
- Ensure Google Cloud credentials are available in the environment when loading from BigQuery.
- Example session ids used for testing can include "1025914_118" (idh) and "1025914_383" (no_idh).
Creates a vertex json for a given session id to test inference.
#1025914_118 idh
#1025914_383 no_idh
"""

import pandas as pd
from google.cloud import bigquery
import json
import idh.gcp.bigquery.tables as bq_tables
import idh.data.utils as data_utils
import idh.data.preprocess as data_prepro
import idh.gcp.bigquery.query as bq_query
import argparse
from typing import Optional
import idh.data.session as session_data


def create_vertex_json(
    project_id: str,
    session_id: str,
    csv_path: Optional[str],
    save_path: str,
) -> None:
    """
    Build and persist a Vertex AI-compatible JSON payload for a single patient session.

    Parameters
    ----------
    project_id : str
        Google Cloud project id used to initialise the BigQuery client when loading
        session data from BigQuery (used when csv_path is None).
    session_id : str
        Session identifier to convert to a Vertex JSON payload (e.g. "1025914_383").
    csv_path : Optional[str]
        Optional path to a CSV file containing the session data. If provided, data is
        loaded from this CSV instead of BigQuery.
    save_path : str
        Path where the generated Vertex JSON payload will be written. If the file
        exists it will be overwritten.

    Returns
    -------
    None
        The function writes the payload to disk at save_path and returns nothing.

    Raises
    ------
    RuntimeError
        If session data cannot be loaded (propagates underlying errors from loaders).
    """
    if csv_path:
        df: pd.DataFrame = session_data.load_from_csv(csv_path)
    else:
        print("No csv path provided. Loading session data from BigQuery...")
        print("Initialising BigQuery client...")
        client = bigquery.Client(project=project_id)
        df: pd.DataFrame = session_data.load_session_data(client, session_id)

    payload: dict = session_data.session_data_to_vertex_json(df)
    data_utils.save_dict_to_json(payload, save_path)


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
        "--csv",
        type=str,
        default=None,
        help="The session id to convert to a vertex json.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="payload.json",
        help="The path to save the vertex json.",
    )
    args = parser.parse_args()
    create_vertex_json(args.project_id, args.session_id, args.csv, args.save_path)
