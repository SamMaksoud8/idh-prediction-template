"""Configuration dataclasses and helpers for the IDH prediction project."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


# ---------- dataclasses (unchanged) ----------
@dataclass
class BigQueryConfig:
    features_dataset: str = ""
    patient_demographics: str = ""
    registration_data: str = ""
    real_time_machine_data: str = ""
    sessionized_machine_data: str = ""


@dataclass
class StorageConfig:
    bucket: str = ""
    csv_prefix: str = ""
    parquet_prefix: str = ""


@dataclass
class ModelConfig:
    name: str = ""
    endpoint_name: str = ""
    model_endpoint: str = ""
    min_replicas: int = 1
    max_replicas: int = 3
    session_window: int = 12
    rolling_window: int = 3
    interval_time: int = 15
    prediction_intervals: int = 5
    idh_threshold: float = 90.0
    docker_image_uri: str = ""
    machine_type: str = "n1-standard-4"
    features: List[str] = field(default_factory=list)


@dataclass
class AppConfig:
    project_name: str = ""
    region: str = ""
    dataset_name: str = ""
    BigQuery: BigQueryConfig = field(default_factory=BigQueryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AppConfig":
        if data is None:
            data = {}
        bq = BigQueryConfig(**data.get("BigQuery", {}))
        storage = StorageConfig(**data.get("storage", {}))
        model = ModelConfig(**data.get("model", {}))
        return AppConfig(
            project_name=data.get("project_name", ""),
            region=data.get("region", ""),
            dataset_name=data.get("dataset_name", ""),
            BigQuery=bq,
            storage=storage,
            model=model,
        )


# ---------- helpers ----------
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-(.*?)|)\}")


def _expand_env(text: str, env: Dict[str, str]) -> str:
    """
    Expand ${VAR} and ${VAR:-default} using provided env (no shell eval).
    Empty-string VAR counts as 'set' (no default), matching shell behavior.
    """

    def repl(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        val = env.get(var)
        if val is None:
            return default if default is not None else ""
        return val

    return _ENV_PATTERN.sub(repl, text)


def load_config(path: Optional[Path] = None, env_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from YAML and optional .env overrides.
    - Loads .env FIRST so env vars are available.
    - Expands ${VAR} and ${VAR:-default} in the YAML text before parsing.
    """
    config_path = (
        Path(path) if path is not None else Path(__file__).parent.parent.parent / "config.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Load .env first (so expansions can see variables)
    resolved_env_path = Path(env_path) if env_path is not None else config_path.parent / ".env"
    if resolved_env_path.exists():
        load_dotenv(dotenv_path=str(resolved_env_path), override=False)

    raw = config_path.read_text(encoding="utf-8")
    expanded = _expand_env(raw, os.environ)

    cfg = yaml.safe_load(expanded) or {}
    app_cfg = AppConfig.from_dict(cfg)

    # Optional explicit overrides via environment variables
    v = os.getenv("MODEL_ENDPOINT_NAME")
    if v is not None:
        app_cfg.model.endpoint_name = v

    model_endpoint_env = os.getenv("MODEL_ENDPOINT")
    model_endpoint_yaml = (cfg.get("model", {}) or {}).get("model_endpoint")
    resolved_endpoint = (
        model_endpoint_env if model_endpoint_env is not None else model_endpoint_yaml
    )
    app_cfg.model.model_endpoint = resolved_endpoint
    app_cfg.model.endpoint = resolved_endpoint  # back-compat if you reference .endpoint elsewhere

    return app_cfg


config = load_config()
