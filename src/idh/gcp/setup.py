"""Convenience functions for provisioning data infrastructure in GCP."""

from __future__ import annotations

from pathlib import Path
from typing import Type

from google.cloud import bigquery
from google.cloud.exceptions import Conflict, GoogleCloudError

import idh.data.fetch as fetch_data
from idh.config import config
from idh.gcp.bigquery.table_manager import TABLES, BigQueryTableManager
import pandas as pd
import os
from idh.gcp.storage.data_schemas import SCHEMAS
from idh.gcp.storage.utils import upload_to_gcs


def create_bigquery_dataset(project_id: str, dataset_id: str, location: str) -> bool:
    """Create a dataset in BigQuery if it does not already exist."""
    print(f"Attempting to create dataset '{dataset_id}' in project '{project_id}'...")

    try:
        client = bigquery.Client(project=project_id)
        full_dataset_id = f"{project_id}.{dataset_id}"
        dataset = bigquery.Dataset(full_dataset_id)
        dataset.location = location
        created_dataset = client.create_dataset(dataset, timeout=30)

        print(
            "✅ Success! Dataset "
            f"'{created_dataset.dataset_id}' created in location '{created_dataset.location}'."
        )
        print(
            "   You can view it at: "
            f"https://console.cloud.google.com/bigquery?project={project_id}&p={project_id}&d={dataset_id}&page=dataset"
        )
        return True

    except Conflict:
        print(
            f"✅ Dataset '{dataset_id}' already exists in project '{project_id}'. No action taken."
        )
        return True

    except GoogleCloudError as exc:
        print(f"❌ An error occurred: {exc}")
        print("   Please check your project ID, permissions, and the chosen location.")
        return False


def convert_csv_to_parquet() -> None:
    """Convert each raw CSV dataset to Parquet using the registered schemas."""
    for schema_cls in SCHEMAS:
        schema_name = getattr(schema_cls, "name", getattr(schema_cls, "__name__", str(schema_cls)))
        print(f"Converting {schema_name} to parquet...")
        schema_cls().to_parquet()


def create_table(
    client: bigquery.Client,
    table_cls: Type[BigQueryTableManager],
    bucket: str,
    prefix: str,
    dataset_id: str,
) -> None:
    """Create and load a table managed by ``table_cls`` from Parquet sources."""
    manager = table_cls(client=client, gs_bucket=bucket, gs_prefix=prefix, dataset_id=dataset_id)
    manager.create_table()
    manager.load_from_gs()


def create_data_tables_from_parquet() -> None:
    """Create raw data tables in BigQuery from Parquet files stored in GCS."""
    client = bigquery.Client()
    bucket = config.storage.bucket
    prefix = config.storage.parquet_prefix
    dataset_id = config.dataset_name
    for table_cls in TABLES:
        create_table(client, table_cls, bucket, prefix, dataset_id)


def load_csv_data_in_gcs(
    bucket: str = config.storage.bucket, prefix: str = config.storage.csv_prefix
) -> None:
    """Download raw CSV files and upload them to ``gs://bucket/prefix``."""
    download_dir = fetch_data.download_raw_csv_files()
    directory = Path(download_dir)

    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix.lower() == ".csv":
            destination_path = f"{prefix}/{entry.name}"
            print(f"Uploading {entry} to gs://{bucket}/{destination_path}...")
            upload_to_gcs(bucket, str(entry), destination_path)

    print("All files uploaded successfully.")
    print(f"Removing temp files from {directory}...")
    fetch_data.delete_temp_dir()
