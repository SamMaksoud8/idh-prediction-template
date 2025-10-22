"""Abstractions for creating and loading BigQuery tables from GCS."""

from __future__ import annotations

from typing import Sequence

from google.cloud import bigquery
from google.cloud.exceptions import Conflict
from idh.config import config


class BigQueryTableManager:
    """Manage creation and loading of a single BigQuery table."""

    def __init__(
        self,
        client: bigquery.Client,
        project_id: str,
        dataset_id: str,
        dataset_name: str,
        schema: Sequence[bigquery.SchemaField],
        gs_bucket: str,
        gs_path: str,
    ) -> None:
        self.client = client
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.table_id = f"{project_id}.{dataset_id}.{dataset_name}"
        self.schema = schema
        self.gs_uri = f"gs://{gs_bucket}/{gs_path}"

    def load_from_gs(self) -> None:
        """Load table data from the configured GCS ``.parquet`` source."""
        print(f"Starting job to load data from {self.gs_uri} into {self.table_id}")

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition="WRITE_TRUNCATE",
        )

        load_job = self.client.load_table_from_uri(
            self.gs_uri, self.table_id, job_config=job_config
        )

        load_job.result()
        print("Load job finished.")

        destination_table = self.client.get_table(self.table_id)
        print(f"Loaded {destination_table.num_rows} rows.")

    def create_table(self) -> None:
        """Create the BigQuery table described by this manager if it is missing."""
        table = bigquery.Table(self.table_id, schema=self.schema)
        try:
            table = self.client.create_table(table)
            print(
                f"Successfully created table: {table.project}.{table.dataset_id}.{table.table_id}"
            )
        except Conflict:
            print(f"Table {self.table_id} already exists.")


class RealTimeMachineDataTable(BigQueryTableManager):
    """Table manager for the ``real-time-machine-data`` table."""

    def __init__(
        self,
        client: bigquery.Client,
        project_id: str = config.project_name,
        dataset_id: str = "raw_data",
        dataset_name: str = "real-time-machine-data",
        gs_bucket: str = config.storage.bucket,
        gs_prefix: str = "raw-data-parquet/vip.parquet",
    ) -> None:
        gs_path = f"{gs_prefix}/vip.parquet"
        schema = [
            bigquery.SchemaField("pid", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("datatime", "DATETIME", mode="NULLABLE"),
            bigquery.SchemaField("measuretime", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("sbp", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("dbp", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("dia_temp_value", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("conductivity", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("uf", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("blood_flow", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("time", "INT64", mode="NULLABLE"),
        ]
        super().__init__(client, project_id, dataset_id, dataset_name, schema, gs_bucket, gs_path)


class PatientDemographicsTable(BigQueryTableManager):
    """Table manager for the ``patient-demographics`` table."""

    def __init__(
        self,
        client: bigquery.Client,
        project_id: str = config.project_name,
        dataset_id: str = "raw_data",
        dataset_name: str = "patient-demographics",
        gs_bucket: str = config.storage.bucket,
        gs_prefix: str = "raw-data-parquet",
    ) -> None:
        gs_path = f"{gs_prefix}/idp.parquet"
        schema = [
            bigquery.SchemaField("pid", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("gender", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("birthday", "DATE", mode="NULLABLE"),
            bigquery.SchemaField("first_dialysis", "DATE", mode="NULLABLE"),
            bigquery.SchemaField("DM", "BOOL", mode="NULLABLE"),
        ]
        super().__init__(client, project_id, dataset_id, dataset_name, schema, gs_bucket, gs_path)


class RegistrationDataTable(BigQueryTableManager):
    """Table manager for the ``registration-data`` table."""

    def __init__(
        self,
        client: bigquery.Client,
        project_id: str = config.project_name,
        dataset_id: str = "raw_data",
        dataset_name: str = "registration-data",
        gs_bucket: str = config.storage.bucket,
        gs_prefix: str = "raw-data-parquet/d1.parquet",
    ) -> None:
        gs_path = f"{gs_prefix}/d1.parquet"
        schema = [
            bigquery.SchemaField("pid", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("keyindate", "DATETIME", mode="NULLABLE"),
            bigquery.SchemaField("dialysisstart", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("dialysisend", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("weightstart", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("weightend", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("dryweight", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("temperature", "FLOAT64", mode="NULLABLE"),
        ]
        super().__init__(client, project_id, dataset_id, dataset_name, schema, gs_bucket, gs_path)


TABLES = [RealTimeMachineDataTable, PatientDemographicsTable, RegistrationDataTable]
