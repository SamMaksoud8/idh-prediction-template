"""Utilities for working with Google Cloud Storage resources."""

from __future__ import annotations

from typing import Optional

from google.api_core import exceptions as api_exceptions
from google.cloud import storage
from google.cloud.exceptions import Conflict, GoogleCloudError


def create_gcs_bucket(bucket_name: str, project_id: str, location: str) -> bool:
    """Create ``bucket_name`` in ``project_id`` if it does not already exist."""
    print(f"Attempting to create bucket '{bucket_name}' in project '{project_id}'...")

    try:
        storage_client = storage.Client(project=project_id)

        existing = storage_client.lookup_bucket(bucket_name)
        if existing:
            loc = getattr(existing, "location", "unknown")
            print(f"ℹ️ Bucket '{bucket_name}' already exists (location: {loc}). Skipping creation.")
            return True

        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        bucket.create(location=location)

        print(f"✅ Success! Bucket '{bucket.name}' created in '{bucket.location}'.")
        print(
            f"   You can view it at: https://console.cloud.google.com/storage/browser/{bucket.name}"
        )
        return True

    except Conflict:
        print(f"ℹ️ Bucket '{bucket_name}' already exists (Conflict on create). Skipping creation.")
        return True

    except GoogleCloudError as exc:
        print(f"❌ An error occurred: {exc}")
        print("   Please check your project ID and make sure you have the 'Storage Admin' role.")
        return False


def create_gspath(bucket: str, prefix: str) -> str:
    """Return a ``gs://`` URI built from ``bucket`` and ``prefix``."""
    return f"gs://{bucket}/{prefix}"


def upload_to_gcs(
    bucket_name: str,
    source_file_path: str,
    destination_blob_name: str,
    *,
    timeout: Optional[int] = 500,
) -> bool:
    """Upload ``source_file_path`` to ``gs://bucket_name/destination_blob_name``."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print(f"Starting upload of {source_file_path} with a {timeout} second timeout...")

    try:
        blob.upload_from_filename(source_file_path, timeout=timeout)
        print(f"✅ Successfully uploaded {source_file_path} to {destination_blob_name}.")
        return True
    except TimeoutError:
        print(
            f"❌ The upload failed because it exceeded the configured timeout of {timeout} seconds."
        )
        print("   Consider increasing the timeout value or checking your network connection.")
        return False
    except api_exceptions.GoogleAPICallError as exc:
        print(f"❌ An API error occurred during the upload: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - unexpected failure reporting
        print(f"❌ An unexpected error occurred during upload: {exc}")
        return False
