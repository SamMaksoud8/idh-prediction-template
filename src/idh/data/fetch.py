"""Download utilities for obtaining sample dialysis datasets."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import requests


def download_file(url: str, local_filename: str) -> None:
    """Download ``url`` to ``local_filename`` with basic retry diagnostics."""
    print(f"Attempting to download {local_filename}...")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(local_filename, "wb") as file_handle:
                for chunk in response.iter_content(chunk_size=8192):
                    file_handle.write(chunk)
        print(f"✅ Successfully downloaded and saved: {local_filename}")
    except requests.exceptions.RequestException as exc:
        print(f"❌ Failed to download {local_filename}. Error: {exc}")


def download_raw_csv_files() -> Path:
    """Download the raw CSV datasets from Figshare and return the temporary folder."""
    url = "https://figshare.com/articles/dataset/Hemrec_VIP_csv/6260654"
    print(f"Downloading raw files from {url}...")
    files_to_download = {
        "d1.csv": "https://figshare.com/ndownloader/files/15142151",
        "idp.csv": "https://figshare.com/ndownloader/files/15142154",
        "vip.csv": "https://figshare.com/ndownloader/files/15142157",
    }

    temp_dir = Path(__file__).resolve().parent / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Starting batch file download into: {temp_dir} ---")

    for filename, file_url in files_to_download.items():
        local_path = temp_dir / filename
        if local_path.exists():
            print(f"ℹ️ Skipping download for {filename}; file already exists at: {local_path}")
            continue
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            download_file(file_url, str(local_path))
            if local_path.exists():
                print(f"✅ {filename} downloaded after {attempt} attempt(s).")
                break

            print(f"⚠️ Attempt {attempt} failed for {filename}.")
            if attempt < max_attempts:
                wait_seconds = 2**attempt
                print(f"⏳ Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
            else:
                print(f"❌ All {max_attempts} attempts failed for {filename}.")
    return temp_dir


def delete_temp_dir(temp_dir: Optional[str | Path] = None) -> bool:
    """Delete the temporary download directory created by :func:`download_raw_csv_files`."""
    if temp_dir is None:
        temp_dir = Path(__file__).resolve().parent / "temp"
    else:
        temp_dir = Path(temp_dir)

    try:
        temp_dir_resolved = temp_dir.resolve()
    except Exception as exc:
        print(f"❌ Could not resolve temp directory path: {exc}")
        return False

    script_parent = Path(__file__).resolve().parent
    expected = script_parent / "temp"

    if temp_dir_resolved != expected:
        print(f"❌ Refusing to delete directory outside the script folder: {temp_dir_resolved}")
        return False

    if not temp_dir_resolved.exists():
        print(f"ℹ️ Temp directory does not exist: {temp_dir_resolved}")
        return True

    try:
        shutil.rmtree(temp_dir_resolved)
        print(f"🗑️ Successfully deleted temp directory: {temp_dir_resolved}")
        return True
    except Exception as exc:
        print(f"❌ Failed to delete temp directory {temp_dir_resolved}: {exc}")
        return False


if __name__ == "__main__":
    download_raw_csv_files()
    # delete_temp_dir()
