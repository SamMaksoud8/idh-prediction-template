"""Schema-aware dataframe loaders for raw dialysis datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from idh.config import config
from idh.gcp.storage.utils import create_gspath


@dataclass
class DataFrameSchema:
    """Base class that reads a CSV and enforces column types via :attr:`schema`."""

    source_path: str = create_gspath(config.storage.bucket, config.storage.csv_prefix)
    destination_path: str = create_gspath(config.storage.bucket, config.storage.parquet_prefix)
    df: Optional[pd.DataFrame] = None
    nrows: Optional[int] = None
    schema: Dict[str, str] = field(default_factory=dict)

    def read_csv(self) -> pd.DataFrame:
        """Load the CSV for this schema into a dataframe."""
        return pd.read_csv(f"{self.source_path}/{self.name}.csv", nrows=self.nrows)

    @property
    def parquet_fpath(self) -> str:
        """Filesystem path used when writing the dataframe to Parquet."""
        return f"{self.destination_path}/{self.name}.parquet"

    def to_parquet(self) -> None:
        """Serialise :attr:`df` to ``.parquet`` using :func:`pandas.DataFrame.to_parquet`."""
        if self.df is None:
            raise ValueError("Dataframe has not been initialised; call read_csv first.")
        self.df.to_parquet(self.parquet_fpath, index=False)

    def post_process(self) -> None:
        """Hook for subclasses to implement additional cleaning steps."""

    def __post_init__(self) -> None:
        """Load the CSV and apply simple ``astype`` conversions declared in ``schema``."""
        self.df = self.read_csv()
        for column_name, target_type in self.schema.items():
            if column_name in self.df.columns:
                try:
                    self.df[column_name] = self.df[column_name].astype(target_type)
                except (ValueError, TypeError) as exc:
                    raise TypeError(
                        "Failed to convert column "
                        f"'{column_name}' to {target_type}. "
                        "Please check the data."
                    ) from exc
        self.post_process()


@dataclass
class PatientDemographics(DataFrameSchema):
    """Schema definition for the ``idp`` demographics CSV."""

    name: str = "idp"
    schema: Dict[str, str] = field(
        default_factory=lambda: {
            "pid": "int64",
            "gender": "object",
            "birthday": "int64",
            "first_dialysis": "object",
            "DM": "bool",
        }
    )

    def post_process(self) -> None:
        if self.df is None:
            return
        self.df["first_dialysis"] = pd.to_datetime(self.df["first_dialysis"])
        self.df["DM"] = self.df["DM"].astype(bool)


@dataclass
class RegistrationData(DataFrameSchema):
    """Schema definition for the ``d1`` registration CSV."""

    name: str = "d1"
    schema: Dict[str, str] = field(
        default_factory=lambda: {
            "pid": "int64",
            "keyindate": "object",
            "dialysisstart": "object",
            "dialysisend": "object",
            "weightstart": "float64",
            "weightend": "float64",
            "dryweight": "float64",
            "temperature": "float64",
        }
    )

    def post_process(self) -> None:
        if self.df is None:
            return
        self.df["keyindate"] = pd.to_datetime(self.df["keyindate"], errors="coerce")

        dialysisstart_str = self.df["dialysisstart"].astype(str)
        dialysisend_str = self.df["dialysisend"].astype(str)

        start_str_combined = self.df["keyindate"].dt.strftime("%Y-%m-%d") + " " + dialysisstart_str
        self.df["dialysisstart"] = pd.to_datetime(start_str_combined, errors="coerce")

        mask_24h = dialysisend_str.str.startswith("24:", na=False)
        end_str_standard = self.df["keyindate"].dt.strftime("%Y-%m-%d") + " " + dialysisend_str
        final_dialysisend_dt = pd.to_datetime(end_str_standard, errors="coerce")

        if mask_24h.any():
            new_date = self.df.loc[mask_24h, "keyindate"] + pd.Timedelta(days=1)
            new_time = "00:" + dialysisend_str[mask_24h].str[3:]
            corrected_end_dt = pd.to_datetime(
                new_date.dt.strftime("%Y-%m-%d") + " " + new_time, errors="coerce"
            )
            final_dialysisend_dt.loc[mask_24h] = corrected_end_dt

        self.df["dialysisend"] = final_dialysisend_dt


@dataclass
class RealTimeMachineData(DataFrameSchema):
    """Schema definition for the ``vip`` machine telemetry CSV."""

    name: str = "vip"
    schema: Dict[str, str] = field(
        default_factory=lambda: {
            "pid": "int64",
            "datatime": "datetime64[ns]",
            "measuretime": "int64",
            "sbp": "int64",
            "dbp": "int64",
            "dia_temp_value": "float64",
            "conductivity": "float64",
            "uf": "float64",
            "blood_flow": "float64",
            "time": "int64",
        }
    )

    def post_process(self) -> None:
        if self.df is None:
            return
        self.df["datatime"] = pd.to_datetime(self.df["datatime"])


SCHEMAS = [PatientDemographics, RegistrationData, RealTimeMachineData]
