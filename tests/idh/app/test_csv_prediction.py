import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


# Stub gradio before importing the module to avoid launching UI
class DummyBlocks:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def launch(self, *args, **kwargs):
        pass


class DummyComponent:
    def __init__(self, *args, **kwargs):
        pass

    def click(self, *args, **kwargs):
        pass


gr_module = types.ModuleType("gradio")
gr_module.Blocks = DummyBlocks
gr_module.Markdown = DummyComponent
gr_module.Row = DummyBlocks
gr_module.Column = DummyBlocks
gr_module.File = DummyComponent
gr_module.Button = DummyComponent
gr_module.Label = DummyComponent
gr_module.Plot = DummyComponent
gr_module.themes = SimpleNamespace(Soft=lambda: None)

sys.modules.setdefault("gradio", gr_module)

px_module = types.ModuleType("plotly.express")


def fake_line(*args, **kwargs):
    return SimpleNamespace(
        update_traces=lambda *a, **k: None,
        update_layout=lambda *a, **k: None,
        update_xaxes=lambda *a, **k: None,
        update_yaxes=lambda *a, **k: None,
    )


px_module.line = fake_line
sys.modules.setdefault("plotly.express", px_module)

import importlib

csv_prediction = importlib.import_module("idh.app.csv_prediction")


def test_resolve():
    preds = [{"predicted_label": "0"}, {"predicted_label": "1"}]
    assert csv_prediction.resolve(preds) is True
    assert csv_prediction.resolve([{"predicted_label": "0"}]) is False


def test_create_sbp_plot_missing_columns():
    df = pd.DataFrame({"a": [1]})
    assert csv_prediction.create_sbp_plot(df) is None


def test_create_sbp_plot_generates_figure():
    df = pd.DataFrame(
        {"datatime": pd.date_range("2024-01-01", periods=2, freq="H"), "sbp": [100, 105]}
    )
    fig = csv_prediction.create_sbp_plot(df)
    assert fig is not None


def test_predict_from_csv_success(monkeypatch, tmp_path):
    df = pd.DataFrame({"datatime": pd.to_datetime(["2024-01-01"]), "sbp": [100]})
    csv_path = tmp_path / "session.csv"
    df.to_csv(csv_path, index=False)

    fake_file = SimpleNamespace(name=str(csv_path))

    monkeypatch.setattr(csv_prediction.session_data, "load_from_csv", lambda path: df)
    monkeypatch.setattr(
        csv_prediction.session_data, "session_data_to_vertex_json", lambda df: {"instances": [1]}
    )
    monkeypatch.setattr(
        csv_prediction.predict, "prepare_payload_for_inference", lambda payload: ([1], {})
    )
    monkeypatch.setattr(csv_prediction, "create_sbp_plot", lambda df: "chart")
    monkeypatch.setattr(
        csv_prediction.predict,
        "predict",
        lambda project, region, endpoint, instances, parameters: SimpleNamespace(
            predictions=[{"predicted_label": "0"}]
        ),
    )

    message, chart = csv_prediction.predict_from_csv(fake_file)
    assert message == "✅ Low IDH Risk"
    assert chart == "chart"


def test_predict_from_csv_errors(monkeypatch):
    class DummyFile:
        name = "file.txt"

    assert "Invalid" in csv_prediction.predict_from_csv(DummyFile())
    assert "Please upload" in csv_prediction.predict_from_csv(None)
