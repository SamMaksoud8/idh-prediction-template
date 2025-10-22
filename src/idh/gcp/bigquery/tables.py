import pandas as pd
from idh.config import config
from google.cloud import bigquery
import time
from google.api_core import exceptions as gcp_exceptions


def features_table_id() -> str:
    """Get the fully-qualified table ID for the features table.

    Returns:
        Fully-qualified table id in the form "project.dataset.table".
    """
    project_id = config.project_name
    dataset = config.dataset_name
    table_name = config.BigQuery.features_dataset
    return f"{project_id}.{dataset}.{table_name}"


def sessionized_machine_data_table_id() -> str:
    """Get the fully-qualified table ID for the sessionized machine data table.

    Returns:
        Fully-qualified table id in the form "project.dataset.table".
    """
    project_id = config.project_name
    dataset = config.dataset_name
    table_name = config.BigQuery.sessionized_machine_data
    return f"{project_id}.{dataset}.{table_name}"


def real_time_machine_data_table_id() -> str:
    """Get the fully-qualified table ID for the real-time machine data table.

    Returns:
        Fully-qualified table id in the form "project.dataset.table".
    """
    project_id = config.project_name
    dataset = config.dataset_name
    table_name = config.BigQuery.real_time_machine_data
    return f"{project_id}.{dataset}.{table_name}"


def registration_data_table_id() -> str:
    """Get the fully-qualified table ID for the registration data table.

    Returns:
        Fully-qualified table id in the form "project.dataset.table".
    """
    project_id = config.project_name
    dataset = config.dataset_name
    table_name = config.BigQuery.registration_data
    return f"{project_id}.{dataset}.{table_name}"


def patient_demographics_table_id() -> str:
    """Get the fully-qualified table ID for the patient demographics table.

    Returns:
        Fully-qualified table id in the form "project.dataset.table".
    """
    project_id = config.project_name
    dataset = config.dataset_name
    table_name = config.BigQuery.patient_demographics
    return f"{project_id}.{dataset}.{table_name}"


def check_table_ready(client: bigquery.Client, table_id: str, max_retries: int = 5):
    """
    Checks for the existence and accessibility of a BigQuery table with
    a retry mechanism (exponential backoff) to account for eventual consistency.

    Args:
        client: The initialized BigQuery client.
        table_id: The full ID of the table (e.g., 'project.dataset.table').
        max_retries: The maximum number of times to retry the check.

    Returns:
        True if the table is successfully accessed, False otherwise.
    """
    # The initial delay (in seconds). This will be squared (exponentiated)
    # for each subsequent retry.
    base_delay_seconds = 4

    print(f"Checking for table: {table_id}")

    for attempt in range(max_retries):
        try:
            # Attempt to retrieve the table metadata
            client.get_table(table_id)
            print(f"✅ Success! Table '{table_id}' is accessible after {attempt + 1} attempt(s).")
            return True

        except gcp_exceptions.NotFound:
            # Calculate the wait time: 2, 4, 8, 16, 32... seconds
            wait_time = base_delay_seconds ** (attempt + 1)
            print(
                f"⚠️ Attempt {attempt + 1} failed (Table Not Found). Waiting {wait_time} seconds..."
            )
            time.sleep(wait_time)

        except Exception as e:
            # Handle other potential errors (e.g., permissions, API issues)
            print(f"🛑 An unexpected error occurred on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                # Only wait if there are more retries left
                time.sleep(base_delay_seconds ** (attempt + 1))
            else:
                break  # Exit the loop if max retries reached

    print(f"❌ Failure! Table '{table_id}' was not accessible after {max_retries} attempts.")
    return False


def load_table_from_bigquery(client: bigquery.Client, table_id: str) -> pd.DataFrame:
    """Loads a full BigQuery table into a pandas DataFrame.

    Args:
        client: BigQuery client (e.g. google.cloud.bigquery.Client).
        table_id: Fully-qualified table id: project.dataset.table

    Returns:
        pd.DataFrame: Loaded table (empty DataFrame on error).
    """
    try:
        print(f"Loading data from {table_id}...")
        df: pd.DataFrame = client.query(f"SELECT * FROM `{table_id}`").to_dataframe()
        print(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"An error occurred while loading {table_id}: {e}")
        # Return an empty DataFrame with expected columns on failure
        # You might want to adjust this based on your needs
        return pd.DataFrame()


def get_table_id(table_name: str) -> str:
    """Build a fully-qualified BigQuery table identifier.

    Args:
        table_name: Table name (without dataset/project).

    Returns:
        Fully-qualified table id in the form "project.dataset.table".
    """
    project_id = config.project_name
    dataset = config.dataset_name
    return f"{project_id}.{dataset}.{table_name}"


def load_real_time_machine_data(client: bigquery.Client) -> pd.DataFrame:
    """Load the real-time machine data table from BigQuery.

    Args:
        client: BigQuery client (google.cloud.bigquery.Client).

    Returns:
        pd.DataFrame: Loaded table as a pandas DataFrame. Returns an empty
        DataFrame on error.
    """
    return load_table_from_bigquery(client, real_time_machine_data_table_id())


def load_registration_data(client: bigquery.Client) -> pd.DataFrame:
    """Load the registration_data table from BigQuery.

    Args:
        client: BigQuery client (google.cloud.bigquery.Client).

    Returns:
        pd.DataFrame: Loaded table as a pandas DataFrame. Returns an empty
        DataFrame on error.
    """
    return load_table_from_bigquery(client, registration_data_table_id())


def load_patient_demographics(client: bigquery.Client) -> pd.DataFrame:
    """Load the patient_demographics table from BigQuery.

    Args:
        client: BigQuery client (google.cloud.bigquery.Client).

    Returns:
        pd.DataFrame: Loaded table as a pandas DataFrame. Returns an empty
        DataFrame on error.
    """
    return load_table_from_bigquery(client, patient_demographics_table_id())


def load_sessionized_machine_data(client: bigquery.Client) -> pd.DataFrame:
    """Load the sessionized_machine_data table from BigQuery.

    Args:
        client: BigQuery client (google.cloud.bigquery.Client).

    Returns:
        pd.DataFrame: Loaded table as a pandas DataFrame. Returns an empty
        DataFrame on error.
    """
    return load_table_from_bigquery(client, sessionized_machine_data_table_id())


if __name__ == "__main__":
    client = bigquery.Client(project=config.project_name)
    print(load_patient_demographics(client).head())
