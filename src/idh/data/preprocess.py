"""Utilities for transforming dialysis session data into model-ready features."""

from __future__ import annotations

import pandas as pd


def get_session_start_time(df: pd.DataFrame) -> pd.Series:
    """Return the earliest measurement timestamp for each session in ``df``.

    Parameters
    ----------
    df:
        Raw session dataframe that includes ``session_id`` and ``datatime`` columns.

    Returns
    -------
    pandas.Series
        A series aligned with ``df`` indicating the start timestamp of each session.
    """
    df = df.copy()
    return df.groupby("session_id")["datatime"].transform("min")


def sessionize_machine_data(df: pd.DataFrame, delta: int = 12) -> pd.DataFrame:
    """Assign ``session_id`` and ``session_start_ts`` columns to machine data rows.

    Parameters
    ----------
    df:
        DataFrame containing at least ``pid`` and ``datatime`` columns.
    delta:
        Session split threshold in hours. A new session starts when the gap between
        consecutive measurements for the same ``pid`` exceeds this value.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with session metadata columns added.
    """
    df = df.copy()

    # Sort values to ensure correct sessionization
    df = df.sort_values(by=["pid", "datatime"]).reset_index(drop=True)

    # Calculate time difference between consecutive records for each patient
    time_diff_hours = df.groupby("pid")["datatime"].diff().dt.total_seconds() / 3600

    # A new session starts if the gap is > ``delta`` hours or if it's the first record
    df["is_new_session"] = (time_diff_hours > delta) | (time_diff_hours.isnull())

    # Create a unique session ID by taking a cumulative sum of new session flags
    session_group = df.groupby("pid")["is_new_session"].cumsum()

    df["session_id"] = df["pid"].astype(str) + "_" + session_group.astype(str)

    # Get the start timestamp for each session
    df["session_start_ts"] = get_session_start_time(df)

    return df


def merge_machine_and_rego_data(df_machine: pd.DataFrame, df_reg: pd.DataFrame) -> pd.DataFrame:
    """Combine machine telemetry with registration metadata.

    Parameters
    ----------
    df_machine:
        Sessionized machine dataframe containing ``pid`` and ``session_start_ts``.
    df_reg:
        Registration dataframe that includes ``pid`` and ``keyindate`` columns.

    Returns
    -------
    pandas.DataFrame
        Joined dataset containing session measurements enriched with registration
        attributes.
    """
    df_machine = df_machine.copy()
    df_reg = df_reg.copy()
    df_machine = df_machine.reset_index(drop=True)
    df_reg = df_reg.reset_index(drop=True)

    # Prepare registration data for joining
    df_reg["keyindate_ts"] = pd.to_datetime(df_reg["keyindate"] / 1000, unit="us")
    df_reg["session_date"] = df_reg["keyindate_ts"].dt.date

    # Prepare machine data for joining
    df_machine["session_date"] = df_machine["session_start_ts"].dt.date

    # Join machine data with registration data on Patient ID and the session date
    return pd.merge(df_machine, df_reg, on=["pid", "session_date"], how="inner")


def merge_patient_demographics(
    df: pd.DataFrame, patient_demographics_df: pd.DataFrame
) -> pd.DataFrame:
    """Add demographic attributes to an aggregated machine dataset.

    Parameters
    ----------
    df:
        DataFrame produced by :func:`merge_machine_and_rego_data`.
    patient_demographics_df:
        Demographic dataframe containing ``pid`` information.

    Returns
    -------
    pandas.DataFrame
        Dataset with merged demographic details and normalized timestamp columns.
    """
    df = df.copy()
    patient_demographics_df = patient_demographics_df.copy()
    df = pd.merge(df, patient_demographics_df, on="pid", how="inner")
    df["first_dialysis_ts"] = pd.to_datetime(df["first_dialysis"] / 1000, unit="us")
    return df


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dialysis telemetry into 15-minute feature bins for inference.

    Parameters
    ----------
    df:
        Combined dataset containing machine, registration, and demographic data for
        multiple sessions. Must include ``datatime``, ``pid`` and ``session_id``.

    Returns
    -------
    pandas.DataFrame
        Aggregated dataset with engineered features and target label for modeling.
    """
    df = df.copy()

    # Bin data into 15-minute intervals
    df["time_bin"] = df["datatime"].dt.floor("15min")

    # Perform aggregations
    aggregations = {
        "sbp": ["mean", "min", "std"],
        "dbp": "mean",
        "dia_temp_value": "mean",
        "conductivity": "mean",
        "uf": "mean",
        "blood_flow": "mean",
        "weightstart": "first",
        "dryweight": "first",
        "gender": "first",
        "birthday": "first",
        "DM": "first",
        "session_start_ts": "first",
        "first_dialysis_ts": "first",
    }
    df_binned = df.groupby(["pid", "session_id", "time_bin"]).agg(aggregations).reset_index()

    # Flatten the multi-level column names after aggregation
    df_binned.columns = ["_".join(col).strip("_") for col in df_binned.columns.values]
    df_binned = df_binned.rename(
        columns={
            "sbp_mean": "avg_sbp",
            "sbp_min": "min_sbp",
            "sbp_std": "stddev_sbp",
            "dbp_mean": "avg_dbp",
            "dia_temp_value_mean": "avg_dia_temp",
            "conductivity_mean": "avg_conductivity",
            "uf_mean": "avg_uf_rate",
            "blood_flow_mean": "avg_blood_flow",
            "weightstart_first": "Weight_start",
            "dryweight_first": "Dry_weight",
            "gender_first": "gender",
            "birthday_first": "birthday",
            "DM_first": "DM",
            "session_start_ts_first": "session_start_ts",
            "first_dialysis_ts_first": "first_dialysis_ts",
        }
    )

    # Add Static and Session-Level Features
    df_binned["age_at_session"] = df_binned["session_start_ts"].dt.year - df_binned["birthday"]

    # Ensure timezone consistency before subtraction to avoid errors
    if (
        df_binned["session_start_ts"].dt.tz is not None
        and df_binned["first_dialysis_ts"].dt.tz is None
    ):
        df_binned["first_dialysis_ts"] = df_binned["first_dialysis_ts"].dt.tz_localize("UTC")

    df_binned["dialysis_vintage_years"] = (
        df_binned["session_start_ts"] - df_binned["first_dialysis_ts"]
    ).dt.days / 365.25
    df_binned["fluid_to_remove"] = df_binned["Weight_start"] - df_binned["Dry_weight"]
    df_binned["minutes_into_session"] = (
        df_binned["time_bin"] - df_binned["session_start_ts"]
    ).dt.total_seconds() / 60

    # Add Window Features and Target Label
    df_binned = df_binned.sort_values(["session_id", "time_bin"]).reset_index(drop=True)

    grouped = df_binned.groupby("session_id")

    # Lag and Trend features
    df_binned["lag_1_avg_sbp"] = grouped["avg_sbp"].shift(1)
    df_binned["lag_1_avg_uf_rate"] = grouped["avg_uf_rate"].shift(1)
    df_binned["trend_1_sbp"] = df_binned["avg_sbp"] - df_binned["lag_1_avg_sbp"]
    df_binned["trend_1_conductivity"] = df_binned["avg_conductivity"] - grouped[
        "avg_conductivity"
    ].shift(1)

    # Rolling features (window=4 corresponds to current row + 3 preceding)
    df_binned["rolling_avg_sbp"] = (
        grouped["avg_sbp"].rolling(window=4, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df_binned["rolling_max_sbp"] = (
        grouped["avg_sbp"].rolling(window=4, min_periods=1).max().reset_index(level=0, drop=True)
    )
    df_binned["rolling_stddev_sbp"] = (
        grouped["avg_sbp"].rolling(window=4, min_periods=1).std().reset_index(level=0, drop=True)
    )

    # We are predicting if hypotension (min_sbp < 90) will occur in the
    # next 15 to 75 minutes (which is 1 to 5 time bins from the current one).
    hypotensive_event_flag = (df_binned["min_sbp"] < 90).astype(int)

    # To look forward, we shift the data *backwards* and then apply a rolling window.
    # This checks the next 5 rows for an event.
    future_events = (
        grouped[hypotensive_event_flag.name].shift(-5).rolling(window=5, min_periods=1).sum()
    )

    # The label is 1 if any event occurred in the future window, otherwise 0.
    df_binned["label"] = (future_events > 0).astype(int).fillna(0).reset_index(level=0, drop=True)

    # Final Selection of Columns
    final_columns = [
        "pid",
        "session_id",
        "DM",
        "age_at_session",
        "avg_blood_flow",
        "avg_conductivity",
        "avg_dbp",
        "avg_dia_temp",
        "avg_sbp",
        "avg_uf_rate",
        "dialysis_vintage_years",
        "fluid_to_remove",
        "gender",
        "lag_1_avg_sbp",
        "lag_1_avg_uf_rate",
        "min_sbp",
        "minutes_into_session",
        "rolling_avg_sbp",
        "rolling_max_sbp",
        "rolling_stddev_sbp",
        "stddev_sbp",
        "trend_1_conductivity",
        "trend_1_sbp",
        "label",
    ]

    # Ensure all required columns exist before returning
    return df_binned[final_columns]
