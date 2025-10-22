import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError

from idh.config import config
from idh.gcp.bigquery.tables import (
    features_table_id,
    patient_demographics_table_id,
    real_time_machine_data_table_id,
    registration_data_table_id,
    sessionized_machine_data_table_id,
)


def run_query(client: bigquery.Client, sql: str, wait: bool = True) -> bigquery.job.QueryJob:
    """
    Execute a BigQuery SQL query and optionally wait for completion.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        An initialized BigQuery client.
    sql : str
        The SQL statement to execute.
    wait : bool, optional
        If True (default), block until the query job completes.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob object representing the launched query. If `wait` is True,
        the returned job has finished.

    Raises
    ------
    google.cloud.exceptions.GoogleCloudError
        Propagates errors raised by the BigQuery client when submitting or running the job.
    """
    job = client.query(sql)  # Make an API request.
    if wait:
        job.result()  # Waits for the query to finish.
    return job


def export_model_to_gcs(
    client: bigquery.Client,
    project_id: str,
    dataset_name: str,
    model_name: str,
    bucket: str,
) -> bigquery.job.QueryJob:
    """
    Export a BigQuery ML model to a Google Cloud Storage bucket.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        An initialized BigQuery client.
    project_id : str
        GCP project ID containing the model.
    dataset_name : str
        BigQuery dataset containing the model.
    model_name : str
        Name of the BigQuery ML model to export.
    bucket : str
        Destination GCS bucket name (without the "gs://" prefix). The model
        artifacts will be exported to gs://{bucket}/model-artifacts/.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob object for the EXPORT MODEL statement. The job may still
        be running when returned (consistent with run_query behavior).

    Raises
    ------
    google.cloud.exceptions.GoogleCloudError
        Propagates errors from the BigQuery client when submitting the export job.
    """
    print(
        f"Starting model deployment for {model_name} to bucket gs://{bucket}/model-artifacts/ ..."
    )
    sql_query = f"""
    EXPORT MODEL `{project_id}.{dataset_name}.{model_name}`
    OPTIONS(URI='gs://{bucket}/model-artifacts/')
    """
    return run_query(client, sql_query)


def export_model_to_gcs_from_config(client: bigquery.Client | None = None) -> bigquery.job.QueryJob:
    """
    Export the model specified in the project config to a Google Cloud Storage bucket.

    This convenience wrapper reads project, dataset, model and bucket settings from
    the global `config` and calls export_model_to_gcs.

    Parameters
    ----------
    client : google.cloud.bigquery.Client | None, optional
        Optional BigQuery client. If None, a client is created using
        config.project_name.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the EXPORT MODEL statement. The job may still be running
        when returned.
    """
    project_id: str = config.project_name
    dataset_name: str = config.dataset_name
    model_name: str = config.model.name
    bucket: str = config.storage.bucket

    if client is None:
        client = bigquery.Client(project=project_id)

    return export_model_to_gcs(
        client=client,
        project_id=project_id,
        dataset_name=dataset_name,
        model_name=model_name,
        bucket=bucket,
    )


def features_engineering(
    client: bigquery.Client,
    features_table: str,
    registration_data_table: str,
    patient_demographics_table: str,
    sessionized_machine_data_table: str,
    interval_window: int,
    rolling_window: int,
    prediction_intervals: int,
    idh_threshold: float,
) -> bigquery.job.QueryJob:
    """
    Create or replace a BigQuery table with engineered features for model training.

    The function builds a multi-step SQL pipeline that:
      - Normalizes and timestamps sessionized machine data
      - Joins registration and demographic information
      - Bins measurements into fixed time intervals
      - Computes static/session-level features
      - Computes lag, trend, rolling-window statistics and a future hypotension target label
      - Splits rows into TRAIN/TEST by a stable hash of session_id

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        Initialized BigQuery client used to run the CREATE OR REPLACE TABLE statement.
    features_table : str
        Fully-qualified destination table name (project.dataset.table) for engineered features.
    registration_data_table : str
        Fully-qualified BigQuery table containing registration data.
    patient_demographics_table : str
        Fully-qualified BigQuery table containing patient demographics.
    sessionized_machine_data_table : str
        Fully-qualified source table containing sessionized machine data.
    interval_window : int
        Time bin size in minutes (e.g. 15 for 15-minute bins).
    rolling_window : int
        Number of preceding rows to include for rolling-window calculations.
    prediction_intervals : int
        Number of following intervals to inspect when computing the hypotension target.
    idh_threshold : float
        Systolic BP threshold to flag hypotension events.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the CREATE OR REPLACE TABLE statement. The job may still be running
        depending on run_query's wait behavior.

    Raises
    ------
    google.cloud.exceptions.GoogleCloudError
        Propagates errors raised by the BigQuery client when submitting or running the job.
    """
    sql_query = f"""
        CREATE OR REPLACE TABLE `{features_table}` AS (
            WITH
            -- ====================================================================================
            -- Step 1: Reference pre-sessionized machine data (normalize cols, keep timestamp)
            -- ====================================================================================
            SessionizedMachineData AS (
            SELECT
                CAST(pid AS INT64) AS Pid,
                CAST(datatime AS TIMESTAMP) AS measurement_timestamp,  -- <-- use as-is (TIMESTAMP)
                sbp, dbp, dia_temp_value, conductivity, uf, blood_flow,
                is_new_session,
                session_id,
                -- Prefer provided session_start_ts; if null, fallback to min(measurement_timestamp)
                COALESCE(
                session_start_ts,
                MIN(CAST(datatime AS TIMESTAMP)) OVER (PARTITION BY session_id)
                ) AS session_start_ts
            FROM `{sessionized_machine_data_table}`
            ),

            -- ====================================================================================
            -- Step 2: Combine with registration and demographic data using the correct date logic
            -- ====================================================================================
            CombinedData AS (
            SELECT
                m.*,
                r.weightstart,
                r.dryweight,
                p.gender,
                p.birthday,
                p.DM,
                SAFE.TIMESTAMP_MICROS(CAST(p.first_dialysis / 1000 AS INT64)) AS first_dialysis_ts
            FROM SessionizedMachineData m
            JOIN `{registration_data_table}` AS r
                ON m.Pid = r.Pid
            AND DATE(m.session_start_ts) = DATE(SAFE.TIMESTAMP_MICROS(CAST(r.keyindate / 1000 AS INT64)))
            JOIN `{patient_demographics_table}` AS p
                ON m.Pid = p.Pid
            ),

            -- ====================================================================================
            -- Step 3: Bin the data into 15-minute intervals for each session.
            -- ====================================================================================
            TimeBinnedData AS (
            SELECT
                Pid, session_id,
                TIMESTAMP_SECONDS(DIV(UNIX_SECONDS(measurement_timestamp), {interval_window} * 60) * ({interval_window} * 60)) AS time_bin,
                ANY_VALUE(session_start_ts) AS session_start_ts,
                AVG(sbp) AS avg_sbp, MIN(sbp) AS min_sbp, STDDEV(sbp) AS stddev_sbp,
                AVG(dbp) AS avg_dbp, AVG(dia_temp_value) AS avg_dia_temp, AVG(conductivity) AS avg_conductivity,
                AVG(uf) AS avg_uf_rate, AVG(blood_flow) AS avg_blood_flow,
                ANY_VALUE(weightstart) AS Weight_start, ANY_VALUE(dryweight) AS Dry_weight,
                ANY_VALUE(gender) AS gender, ANY_VALUE(birthday) AS birthday,
                ANY_VALUE(DM) AS DM, ANY_VALUE(first_dialysis_ts) AS first_dialysis_ts
            FROM CombinedData
            WHERE measurement_timestamp IS NOT NULL
            GROUP BY 1, 2, 3
            ),

            -- ====================================================================================
            -- Step 4: Add static and session-level features.
            -- ====================================================================================
            StaticAndSessionFeatures AS (
            SELECT
                *,
                EXTRACT(YEAR FROM session_start_ts) - birthday AS age_at_session,
                DATE_DIFF(DATE(session_start_ts), DATE(first_dialysis_ts), YEAR) AS dialysis_vintage_years,
                Weight_start - Dry_weight AS fluid_to_remove,
                TIMESTAMP_DIFF(time_bin, session_start_ts, MINUTE) AS minutes_into_session
            FROM TimeBinnedData
            ),

            -- ====================================================================================
            -- Step 5 & 6: Lag, Trend, Rolling, and Target Label features (Partition by session_id)
            -- ====================================================================================
            FinalFeatures AS (
            SELECT
                *,
                LAG(avg_sbp, 1) OVER (PARTITION BY session_id ORDER BY time_bin) AS lag_1_avg_sbp,
                avg_sbp - LAG(avg_sbp, 1) OVER (PARTITION BY session_id ORDER BY time_bin) AS trend_1_sbp,
                LAG(avg_uf_rate, 1) OVER (PARTITION BY session_id ORDER BY time_bin) AS lag_1_avg_uf_rate,
                avg_conductivity - LAG(avg_conductivity, 1) OVER (PARTITION BY session_id ORDER BY time_bin) AS trend_1_conductivity,
                AVG(avg_sbp) OVER (PARTITION BY session_id ORDER BY time_bin ROWS BETWEEN {rolling_window} PRECEDING AND CURRENT ROW) AS rolling_avg_sbp,
                MAX(avg_sbp) OVER (PARTITION BY session_id ORDER BY time_bin ROWS BETWEEN {rolling_window} PRECEDING AND CURRENT ROW) AS rolling_max_sbp,
                STDDEV(avg_sbp) OVER (PARTITION BY session_id ORDER BY time_bin ROWS BETWEEN {rolling_window} PRECEDING AND CURRENT ROW) AS rolling_stddev_sbp,
                MAX(IF(min_sbp < {idh_threshold}, 1, 0)) OVER (PARTITION BY session_id ORDER BY time_bin ROWS BETWEEN 1 FOLLOWING AND {prediction_intervals} FOLLOWING)
                AS hypotension_in_next_15_to_75_min
            FROM StaticAndSessionFeatures
            )

            -- ====================================================================================
            -- Final Selection with the added dataset_split column
            -- ====================================================================================
            SELECT
            CASE
                WHEN MOD(ABS(FARM_FINGERPRINT(session_id)), 10) < 8 THEN 'TRAIN'
                ELSE 'TEST'
            END AS dataset_split,

            Pid, session_id, time_bin,
            age_at_session, dialysis_vintage_years, fluid_to_remove, minutes_into_session,
            gender, DM,
            avg_sbp, min_sbp, stddev_sbp, avg_dbp, avg_dia_temp, avg_conductivity, avg_uf_rate, avg_blood_flow,
            lag_1_avg_sbp, trend_1_sbp,
            lag_1_avg_uf_rate,
            trend_1_conductivity,
            rolling_avg_sbp, rolling_max_sbp, rolling_stddev_sbp,
            COALESCE(hypotension_in_next_15_to_75_min, 0) AS label
            FROM FinalFeatures
            );
        """
    print(f"Starting feature engineering to create/update table `{features_table}`...")
    return run_query(client, sql_query)


def features_engineering_from_config(
    client: bigquery.Client | None = None,
) -> bigquery.job.QueryJob:
    """
    Create or replace the features table using project configuration.

    This convenience wrapper reads project, dataset and model settings from the global
    `config`, constructs fully-qualified table names, creates a BigQuery client if
    none is provided, and calls `features_engineering` to build the features table.

    Parameters
    ----------
    client : google.cloud.bigquery.Client | None, optional
        Optional BigQuery client. If None, a client is created using config.project_name.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the CREATE OR REPLACE TABLE statement. The job may still be running.
    """
    project_id: str = config.project_name

    features_table: str = features_table_id()
    sessionized_table_name: str = sessionized_machine_data_table_id()
    registration_data_table: str = registration_data_table_id()
    patient_demographics_table: str = patient_demographics_table_id()

    rolling_window: int = config.model.rolling_window
    interval_time: int = config.model.interval_time
    prediction_intervals: int = config.model.prediction_intervals
    idh_threshold: float = config.model.idh_threshold

    if client is None:
        client = bigquery.Client(project=project_id)

    return features_engineering(
        client=client,
        features_table=features_table,
        registration_data_table=registration_data_table,
        patient_demographics_table=patient_demographics_table,
        sessionized_machine_data_table=sessionized_table_name,
        rolling_window=rolling_window,
        interval_window=interval_time,
        prediction_intervals=prediction_intervals,
        idh_threshold=idh_threshold,
    )


def evaluate_model(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    model_name: str,
    wait: bool = True,
) -> bigquery.job.QueryJob:
    """
    Evaluate a BigQuery ML model using ML.EVALUATE.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        Initialized BigQuery client.
    project_id : str
        GCP project ID containing the model.
    dataset_id : str
        BigQuery dataset containing the model.
    model_name : str
        Name of the BigQuery ML model to evaluate.
    wait : bool, optional
        If True (default), wait for the query job to complete before returning.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the ML.EVALUATE call. If `wait` is True, the job has finished.
    """
    sql_query = f"""
        SELECT
          *
        FROM
          ML.EVALUATE(MODEL `{project_id}.{dataset_id}.{model_name}`);
    """
    return run_query(client, sql_query, wait=wait)


def train_xgboost_model(
    client: bigquery.Client,
    features: list[str],
    project_id: str,
    dataset_id: str,
    features_name: str,
    model_name: str,
) -> bigquery.job.QueryJob:
    """
    Train a BigQuery ML XGBoost (BOOSTED_TREE_CLASSIFIER) model using a features table.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        Initialized BigQuery client.
    features : list[str]
        List of feature column names to use for training.
    project_id : str
        GCP project ID where the model will be created.
    dataset_id : str
        BigQuery dataset ID where the model and features table reside.
    features_name : str
        Name of the features table containing preprocessed training features.
    model_name : str
        Name to give the created BigQuery ML model.
    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the CREATE OR REPLACE MODEL statement (completed if run_query waits).
    """
    label = "label"
    select_columns = ",\n  ".join(features + [label])

    sql_query = f"""
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.{model_name}`
    OPTIONS(
      MODEL_TYPE='BOOSTED_TREE_CLASSIFIER',
      INPUT_LABEL_COLS=['{label}'],
      AUTO_CLASS_WEIGHTS=TRUE,
      ENABLE_GLOBAL_EXPLAIN=TRUE,
      MODEL_REGISTRY='VERTEX_AI',
      VERTEX_AI_MODEL_ID='{model_name}',
      VERTEX_AI_MODEL_VERSION_ALIASES=['prod', 'initial']
    ) AS
    SELECT
      {select_columns}
    FROM
      `{project_id}.{dataset_id}.{features_name}`
    WHERE
      -- Explicitly tell the model to ONLY train on the 'TRAIN' data
      dataset_split = 'TRAIN';
    """
    print(f"Starting model training for {model_name}...")
    job = run_query(client, sql_query)
    print("Model training complete.")
    print(f"Model saved to {project_id}.{dataset_id}.{model_name}.")
    return job


def get_session_machine_data(client: bigquery.Client, session_id: str) -> pd.DataFrame | None:
    """
    Retrieve sessionized machine data for a given session_id from BigQuery.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        Initialized BigQuery client.
    session_id : str
        The session identifier to filter records by.

    Returns
    -------
    pandas.DataFrame | None
        A DataFrame containing rows for the requested session ordered by datatime,
        an empty DataFrame if the query returns no rows, or None if an error occurred.

    Notes
    -----
    Errors from the BigQuery client are caught and logged; this function returns
    None on error instead of raising.
    """
    try:
        table_id: str = sessionized_machine_data_table_id()

        sql_query: str = f"""
            SELECT *
            FROM `{table_id}`
            WHERE session_id = @session_id
            ORDER BY datatime
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            ]
        )

        print(f"Querying for session_id: {session_id}...")

        query_job: bigquery.job.QueryJob = client.query(sql_query, job_config=job_config)

        results_df: pd.DataFrame = query_job.to_dataframe()

        if results_df.empty:
            print("Query successful, but no data was found for this session_id.")
        else:
            print(f"Successfully retrieved {len(results_df)} rows.")

        return results_df

    except GoogleCloudError as e:
        print(f"An error occurred while running the BigQuery job: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def sessionize_machine_data(
    client: bigquery.Client,
    session_window: int,
    table_name: str,
    sessionized_table_name: str,
) -> bigquery.job.QueryJob:
    """
    Create or replace a sessionized table in BigQuery by grouping records into sessions.

    A new session is started when the gap between consecutive records for the same pid
    is greater than `session_window` hours or when there is no prior record for the pid.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
        Initialized BigQuery client.
    session_window : int
        Session gap threshold in hours. Gaps greater than this start a new session.
    table_name : str
        Fully qualified source table name (project.dataset.table) to read raw records from.
    sessionized_table_name : str
        Fully qualified destination table name (project.dataset.table) to create/replace.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the CREATE OR REPLACE TABLE statement (completed if run_query waits).
    """
    sql_query: str = f"""
        CREATE OR REPLACE TABLE `{sessionized_table_name}` AS (
        WITH
            -- Step 1: Flag the start of each new session
            NewSessionFlags AS (
            SELECT
                pid,
                -- Correctly convert the raw integer to a proper timestamp
                SAFE.TIMESTAMP_MICROS(CAST(datatime / 1000 AS INT64)) AS datatime,
                sbp,
                dbp,
                dia_temp_value,
                conductivity,
                uf,
                blood_flow,
                -- Flag a record as the start of a new session
                CASE
                WHEN
                    LAG(SAFE.TIMESTAMP_MICROS(CAST(datatime / 1000 AS INT64)), 1) OVER (
                    PARTITION BY pid ORDER BY SAFE.TIMESTAMP_MICROS(CAST(datatime / 1000 AS INT64))
                    ) IS NULL
                    THEN 1
                WHEN
                    TIMESTAMP_DIFF(
                    SAFE.TIMESTAMP_MICROS(CAST(datatime / 1000 AS INT64)),
                    LAG(SAFE.TIMESTAMP_MICROS(CAST(datatime / 1000 AS INT64)), 1) OVER (
                        PARTITION BY pid ORDER BY SAFE.TIMESTAMP_MICROS(CAST(datatime / 1000 AS INT64))
                    ),
                    HOUR
                    ) > {session_window}
                    THEN 1
                ELSE 0
                END AS is_new_session
            FROM
                `{table_name}`
            ),

            -- Step 2: Create a unique session ID for each group of records
            SessionIdentifiers AS (
            SELECT
                * ,
                -- The session_id is created by combining pid and a cumulative sum
                CONCAT(
                CAST(pid AS STRING),
                '_',
                CAST(SUM(is_new_session) OVER (PARTITION BY pid ORDER BY datatime) AS STRING)
                ) AS session_id
            FROM
                NewSessionFlags
            )

        -- Step 3: Final Selection and adding the session start time
        SELECT
            * ,
            MIN(datatime) OVER (PARTITION BY session_id) AS session_start_ts
        FROM
            SessionIdentifiers
        );
        """

    print(
        f"Starting the BigQuery job to create the sessionized table `{sessionized_table_name}`..."
    )

    query_job: bigquery.job.QueryJob = run_query(client, sql_query)

    print(f"✅ Success! Table `{sessionized_table_name}` has been created.")
    return query_job


def sessionize_machine_data_from_config(
    client: bigquery.Client | None = None,
) -> bigquery.job.QueryJob:
    """
    Create or replace the sessionized machine data table using settings from the config.

    Parameters
    ----------
    client : google.cloud.bigquery.Client | None
        Optional BigQuery client. If None, a client is created using config.project_name.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the CREATE OR REPLACE TABLE statement created by sessionize_machine_data.
    """
    project_id = config.project_name
    if client is None:
        client = bigquery.Client(project=project_id)

    session_window = config.model.session_window

    table_name = real_time_machine_data_table_id()
    sessionized_table_name = sessionized_machine_data_table_id()

    return sessionize_machine_data(
        client=client,
        session_window=session_window,
        table_name=table_name,
        sessionized_table_name=sessionized_table_name,
    )


def run_model_evaluation_from_config(
    client: bigquery.Client | None = None,
) -> bigquery.job.QueryJob:
    """
    Evaluate the model specified in the project's config using ML.EVALUATE.

    Parameters
    ----------
    client : google.cloud.bigquery.Client | None
        Optional BigQuery client. If None, a client is created using config.project_name.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the ML.EVALUATE call (completed if evaluate_model waits).
    """
    project_id = config.project_name
    if client is None:
        client = bigquery.Client(project=project_id)

    dataset_id = config.dataset_name
    model_name = config.model.name

    return evaluate_model(client, project_id, dataset_id, model_name)


def run_model_training_from_config(client: bigquery.Client | None = None) -> bigquery.job.QueryJob:
    """
    Train an XGBoost (BOOSTED_TREE_CLASSIFIER) model using settings from the project's config.

    Parameters
    ----------
    client : google.cloud.bigquery.Client | None
        Optional BigQuery client. If None, a client is created using config.project_name.

    Returns
    -------
    google.cloud.bigquery.job.QueryJob
        The QueryJob for the CREATE OR REPLACE MODEL statement (completed if run_query waits).
    """
    project_id = config.project_name
    if client is None:
        client = bigquery.Client(project=project_id)

    features = config.model.features
    dataset_id = config.dataset_name
    model_name = config.model.name
    features_name = config.BigQuery.features_dataset
    return train_xgboost_model(
        client=client,
        features=features,
        project_id=project_id,
        dataset_id=dataset_id,
        features_name=features_name,
        model_name=model_name,
    )
