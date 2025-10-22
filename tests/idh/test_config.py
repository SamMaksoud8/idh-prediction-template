import os
from pathlib import Path

import pytest

import idh.config as config_module


def test_app_config_from_dict_defaults():
    app_cfg = config_module.AppConfig.from_dict({})
    assert app_cfg.project_name == ""
    assert app_cfg.BigQuery.features_dataset == ""


def test_load_config_with_env_override(tmp_path, monkeypatch):
    config_data = """
project_name: demo
region: test-region
model:
  endpoint_name: yaml-endpoint
  model_endpoint: yaml-resource
    """
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_data, encoding="utf-8")

    env_path = tmp_path / ".env"
    env_path.write_text(
        "MODEL_ENDPOINT_NAME=env-endpoint\nMODEL_ENDPOINT=env-resource\n", encoding="utf-8"
    )

    monkeypatch.setenv("MODEL_ENDPOINT_NAME", "env-endpoint")
    monkeypatch.setenv("MODEL_ENDPOINT", "env-resource")

    app_cfg = config_module.load_config(cfg_path, env_path)
    assert app_cfg.project_name == "demo"
    assert app_cfg.model.endpoint_name == "env-endpoint"
    assert app_cfg.model.endpoint == "env-resource"


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        config_module.load_config(tmp_path / "missing.yaml")
