"""
Setup script for provisioning cloud storage and BigQuery resources and loading dataset files.
This module provides a small orchestration script intended to be executed as a standalone
utility to prepare Google Cloud Platform resources and ingest data for the idh project.
What it does:
- Creates a Google Cloud Storage (GCS) bucket (name and location taken from idh.config).
- Uploads CSV source data into the GCS bucket.
- Converts the uploaded CSV files to Parquet format in GCS.
- Creates a BigQuery dataset in the configured project and location.
- Creates BigQuery tables based on the generated Parquet files.
Configuration:
- Reads configuration values from idh.config:
    - config.project_name: GCP project id to use for bucket and dataset creation.
    - config.storage.bucket: name of the GCS bucket to create/use.
    - config.region: GCP region/location for bucket and dataset.
    - config.dataset_name: BigQuery dataset id to create/use.
Authentication and permissions:
- Requires Google Cloud credentials with sufficient IAM permissions to create buckets,
    upload objects, create datasets, and create/modify BigQuery tables (e.g., roles/storage.admin,
    roles/bigquery.admin or equivalent).
- Typical authentication methods: Application Default Credentials, gcloud auth login,
    or a service account key provided via environment variables.
Execution:
- Intended to be run as a script (e.g., python scripts/setup_data.py).
- The script prints progress messages and delegates the heavy lifting to helper functions in
    idh.gcp.setup and idh.gcp.storage.utils.
Behavior and idempotency:
- The underlying helper functions are expected to handle existing resources gracefully where
    possible, but callers should be prepared for exceptions if resources already exist or if
    permission issues occur.
- Side effects include creation of GCS bucket, uploaded and converted data files, and
    BigQuery dataset/tables in the target project.
Errors and troubleshooting:
- Errors may arise from invalid configuration, insufficient IAM permissions, quota limits,
    or connectivity issues. Inspect printed messages and underlying exception traces to
    determine the root cause.
"""

from idh.config import config
from idh.gcp.storage.utils import create_gcs_bucket
import idh.gcp.setup as gcp_setup


def setup_data() -> None:
    """
    Prepare GCP resources and load the dataset according to idh.config.

    Steps performed:
    - Create a GCS bucket.
    - Upload CSV source files to the bucket.
    - Convert CSV files in GCS to Parquet.
    - Create a BigQuery dataset.
    - Create BigQuery tables from the Parquet files.

    Uses configuration from idh.config.config (project_name, storage.bucket, region, dataset_name).

    Raises:
        Exception: Propagates exceptions from underlying GCP helper functions on failure.
    """
    project_id: str = config.project_name
    bucket_name: str = config.storage.bucket
    location: str = config.region

    print(
        f"Creating GCS bucket '{bucket_name}' in project '{project_id}' at location '{location}'..."
    )
    create_gcs_bucket(bucket_name, project_id, location)

    print("Downloading CSV data to GCS...")
    gcp_setup.load_csv_data_in_gcs()

    print("Converting CSV data to Parquet...")
    gcp_setup.convert_csv_to_parquet()

    print(
        f"Creating BigQuery dataset '{config.dataset_name}' in project '{project_id}' at location '{location}'..."
    )
    gcp_setup.create_bigquery_dataset(
        project_id=project_id,
        dataset_id=config.dataset_name,
        location=location,
    )

    print("Creating BigQuery tables from Parquet data...")
    gcp_setup.create_data_tables_from_parquet()
    return None


if __name__ == "__main__":
    setup_data()
