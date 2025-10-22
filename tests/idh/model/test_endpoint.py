from types import SimpleNamespace

import idh.model.endpoint as endpoint


def test_check_endpoint_deployed(monkeypatch):
    class FakeEndpoint:
        def __init__(self, endpoint_name=None):
            self.resource_name = endpoint_name
            self.gca_resource = SimpleNamespace(deployed_models=[1])

    original_endpoint = FakeEndpoint("name")

    def fake_endpoint(endpoint_name):
        return FakeEndpoint(endpoint_name)

    monkeypatch.setattr(endpoint, "aiplatform", SimpleNamespace(Endpoint=fake_endpoint))

    assert endpoint.check_endpoint_deployed(original_endpoint) is True


def test_create_vertex_endpoint(monkeypatch):
    created = {}

    class FakeEndpoint:
        @classmethod
        def create(cls, display_name):
            created["display_name"] = display_name
            ep = cls()
            ep.resource_name = "projects/test/locations/us/endpoints/123"
            ep.display_name = display_name
            return ep

    fake_module = SimpleNamespace(
        init=lambda project, location: created.update({"init": (project, location)}),
        Endpoint=FakeEndpoint,
    )
    monkeypatch.setattr(endpoint, "aiplatform", fake_module)

    ep = endpoint.create_vertex_endpoint("proj", "us", "display")
    assert created["init"] == ("proj", "us")
    assert ep.display_name == "display"


def test_deploy_vertex_model(monkeypatch, tmp_path):
    deployed = {}

    class FakeEndpoint:
        def __init__(self, display_name="ep"):
            self.display_name = display_name
            self.resource_name = "projects/test/locations/us/endpoints/123"

        def deploy(self, **kwargs):
            deployed.update(kwargs)

    class FakeModel:
        def __init__(self, model_name):
            self.display_name = model_name

    def fake_init(project=None, location=None):
        deployed["init"] = (project, location)

    fake_module = SimpleNamespace(init=fake_init, Model=FakeModel, Endpoint=FakeEndpoint)
    monkeypatch.setattr(endpoint, "aiplatform", fake_module)

    env_path = tmp_path / ".env"
    ep = FakeEndpoint("endpoint")

    result = endpoint.deploy_vertex_model(
        "proj",
        "us",
        "model",
        ep,
        env_file_path=str(env_path),
        machine_type="type",
        min_replicas=1,
        max_replicas=2,
    )

    assert deployed["init"] == ("proj", "us")
    assert deployed["model"].display_name == "model"
    assert env_path.read_text().strip().startswith("MODEL_ENDPOINT=")
    assert result is ep


def test_deploy_model_from_config(monkeypatch):
    fake_config = SimpleNamespace(
        project_name="proj",
        region="us",
        model=SimpleNamespace(name="model", machine_type="type", min_replicas=1, max_replicas=2),
    )
    monkeypatch.setattr(endpoint, "config", fake_config)

    called = {}

    def fake_deploy_vertex_model(**kwargs):
        called.update(kwargs)
        return "endpoint"

    monkeypatch.setattr(endpoint, "deploy_vertex_model", fake_deploy_vertex_model)

    result = endpoint.deploy_model_from_config("endpoint_obj", env_file_path="file")
    assert result == "endpoint"
    assert called["project_id"] == "proj"
    assert called["endpoint"] == "endpoint_obj"


def test_get_endpoint(monkeypatch):
    class FakeEndpoint:
        def __init__(self, display_name, name):
            self.display_name = display_name
            self.name = name

    endpoints = [FakeEndpoint("match", "resource"), FakeEndpoint("other", "res")]

    fake_module = SimpleNamespace(
        init=lambda project, location: None, Endpoint=SimpleNamespace(list=lambda: endpoints)
    )
    monkeypatch.setattr(endpoint, "aiplatform", fake_module)

    result = endpoint.get_endpoint("proj", "us", "match")
    assert result.name == "resource"
    assert endpoint.get_endpoint("proj", "us", "missing") is None


def test_get_endpoint_id(monkeypatch):
    fake_endpoint = SimpleNamespace(name="resource")
    monkeypatch.setattr(endpoint, "get_endpoint", lambda project, location, name: fake_endpoint)
    assert endpoint.get_endpoint_id("proj", "us", "display") == "resource"
    monkeypatch.setattr(endpoint, "get_endpoint", lambda project, location, name: None)
    assert endpoint.get_endpoint_id("proj", "us", "display") is None


def test_get_endpoint_id_from_config(monkeypatch):
    fake_config = SimpleNamespace(
        model=SimpleNamespace(endpoint=None, endpoint_name="display"),
        project_name="proj",
        region="us",
    )
    monkeypatch.setattr(endpoint, "config", fake_config)
    monkeypatch.setattr(endpoint, "get_endpoint_id", lambda project, location, display: "resolved")

    assert endpoint.get_endpoint_id_from_config() == "resolved"

    fake_config.model.endpoint = "direct"
    monkeypatch.setattr(endpoint, "config", fake_config)
    assert endpoint.get_endpoint_id_from_config() == "direct"
