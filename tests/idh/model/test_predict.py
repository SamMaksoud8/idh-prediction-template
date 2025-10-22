from types import SimpleNamespace
from unittest.mock import MagicMock

import idh.model.predict as predict


def test_prepare_payload_for_inference_defaults():
    instances, parameters = predict.prepare_payload_for_inference(None)
    assert instances == []
    assert parameters == {}

    payload = {"instances": [1], "parameters": {"a": 1}}
    instances, parameters = predict.prepare_payload_for_inference(payload)
    assert instances == [1]
    assert parameters == {"a": 1}


def test_predict_invokes_endpoint(monkeypatch):
    called = {}

    def fake_init(project=None, location=None):
        called["init"] = (project, location)

    class FakeEndpoint:
        def __init__(self, endpoint_name):
            self.endpoint_name = endpoint_name

        def predict(self, instances=None, parameters=None):
            called["predict"] = (instances, parameters, self.endpoint_name)
            return SimpleNamespace(predictions=[{"predicted_label": "0"}])

    monkeypatch.setattr(predict.aiplatform, "init", fake_init)
    monkeypatch.setattr(predict.aiplatform, "Endpoint", FakeEndpoint)

    response = predict.predict("proj", "region", "123", [1], {"param": 1})
    assert called["init"] == ("proj", "region")
    assert called["predict"][0] == [1]
    assert called["predict"][1] == {"param": 1}
    assert "123" in called["predict"][2]
    assert response.predictions == [{"predicted_label": "0"}]


def test_predict_from_config(monkeypatch):
    fake_config = SimpleNamespace(project_name="proj", region="region")
    monkeypatch.setattr(predict, "config", fake_config)
    monkeypatch.setattr(predict, "get_endpoint_id_from_config", lambda: "endpoint")

    captured = {}

    def fake_predict(project_id, region, endpoint_id, instances, parameters):
        captured.update(
            {
                "project_id": project_id,
                "region": region,
                "endpoint_id": endpoint_id,
                "instances": instances,
                "parameters": parameters,
            }
        )
        return "response"

    monkeypatch.setattr(predict, "predict", fake_predict)

    result = predict.predict_from_config([1, 2], {"param": 1})
    assert result == "response"
    assert captured["endpoint_id"] == "endpoint"
    assert captured["instances"] == [1, 2]
