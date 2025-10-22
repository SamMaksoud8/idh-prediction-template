import sys
import types
from pathlib import Path
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _ensure_module(name):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


# Stub google.cloud.bigquery and related modules to avoid installing heavy deps.
_ensure_module("google")
_ensure_module("google.cloud")

bigquery_module = types.ModuleType("google.cloud.bigquery")


class DummyQueryJob:
    def __init__(self, sql=None):
        self.sql = sql
        self.result_called = False

    def result(self):
        self.result_called = True
        return self

    def to_dataframe(self):
        return MagicMock()


class DummyBigQueryClient:
    def __init__(self, project=None):
        self.project = project
        self.queries = []
        self.loaded_tables = {}
        self.created_tables = []
        self.tables = {}

    def query(self, sql):
        self.queries.append(sql)
        job = DummyQueryJob(sql)
        return job

    def load_table_from_uri(self, uri, table_id, job_config=None):
        job = DummyQueryJob(sql=f"LOAD {uri} -> {table_id}")
        self.loaded_tables[table_id] = {
            "uri": uri,
            "job_config": job_config,
        }
        return job

    def get_table(self, table_id):
        return self.tables.get(table_id, SimpleNamespace(num_rows=0))

    def create_dataset(self, dataset, timeout=None):
        dataset_id = getattr(dataset, "dataset_id", None)
        self.created_tables.append(dataset_id)
        return dataset

    def create_table(self, table):
        self.tables[table.table_id] = SimpleNamespace(
            project=self.project,
            dataset_id=table.dataset_id,
            table_id=table.table_id,
        )
        return self.tables[table.table_id]


class DummyLoadJobConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummySchemaField:
    def __init__(self, name, field_type, mode="NULLABLE"):
        self.name = name
        self.field_type = field_type
        self.mode = mode


class DummyDataset:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id.split(".")[-1]
        self.location = None


class DummyTable:
    def __init__(self, table_id, schema=None):
        self.table_id = table_id
        self.schema = schema or []
        parts = table_id.split(".")
        self.project = parts[0] if len(parts) > 0 else None
        self.dataset_id = parts[1] if len(parts) > 1 else None
        self.table_id_only = parts[2] if len(parts) > 2 else None


bigquery_module.Client = DummyBigQueryClient
bigquery_module.LoadJobConfig = DummyLoadJobConfig
bigquery_module.SourceFormat = SimpleNamespace(PARQUET="PARQUET")
bigquery_module.SchemaField = DummySchemaField
bigquery_module.Dataset = DummyDataset
bigquery_module.Table = DummyTable
bigquery_module.job = SimpleNamespace(QueryJob=DummyQueryJob)

sys.modules["google.cloud.bigquery"] = bigquery_module

# google.cloud.exceptions
exceptions_module = types.ModuleType("google.cloud.exceptions")


class GoogleCloudError(Exception):
    pass


class Conflict(Exception):
    pass


exceptions_module.GoogleCloudError = GoogleCloudError
exceptions_module.Conflict = Conflict
sys.modules["google.cloud.exceptions"] = exceptions_module

# google.api_core.exceptions
api_core_exceptions = types.ModuleType("google.api_core.exceptions")


class NotFound(Exception):
    pass


class ApiCoreConflict(Exception):
    pass


api_core_exceptions.NotFound = NotFound
api_core_exceptions.Conflict = ApiCoreConflict
sys.modules["google.api_core.exceptions"] = api_core_exceptions

# google.cloud.storage
storage_module = types.ModuleType("google.cloud.storage")


class DummyBlob:
    def __init__(self, name):
        self.name = name
        self.uploaded_from = None

    def upload_from_filename(self, filename, timeout=None):
        self.uploaded_from = (filename, timeout)


class DummyBucket:
    def __init__(self, name):
        self.name = name
        self.location = "US"
        self._blobs = {}

    def blob(self, name):
        blob = DummyBlob(name)
        self._blobs[name] = blob
        return blob

    def create(self, location=None):
        self.location = location or self.location
        return self


class DummyStorageClient:
    def __init__(self, project=None):
        self.project = project
        self.buckets = {}

    def bucket(self, name):
        bucket = self.buckets.setdefault(name, DummyBucket(name))
        return bucket

    def lookup_bucket(self, name):
        return self.buckets.get(name)

    def create_bucket(self, name, location=None):
        bucket = DummyBucket(name)
        bucket.location = location
        self.buckets[name] = bucket
        return bucket


storage_module.Client = DummyStorageClient
sys.modules["google.cloud.storage"] = storage_module

# google.cloud.aiplatform
ai_module = types.ModuleType("google.cloud.aiplatform")


class DummyModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.display_name = model_name


class DummyEndpoint:
    registry = {}
    init_args = {}

    def __init__(self, endpoint_name=None, display_name=None):
        self.display_name = display_name
        if endpoint_name:
            self.resource_name = endpoint_name
            stored = self.registry.get(endpoint_name)
            if stored:
                self._gca_resource = stored._gca_resource
                self.display_name = stored.display_name
            else:
                self._gca_resource = SimpleNamespace(deployed_models=[])
        else:
            self.resource_name = (
                f"projects/test/locations/test/endpoints/{display_name or 'endpoint'}"
            )
            self._gca_resource = SimpleNamespace(deployed_models=[])
        self.name = self.resource_name
        self.deployed = False

    @property
    def gca_resource(self):
        return self._gca_resource

    def predict(self, instances=None, parameters=None):
        return SimpleNamespace(predictions=[{"predicted_label": "0"}])

    def deploy(
        self,
        model=None,
        deployed_model_display_name=None,
        machine_type=None,
        min_replica_count=None,
        max_replica_count=None,
        traffic_split=None,
    ):
        self.deployed = True
        self._gca_resource.deployed_models = [deployed_model_display_name]

    @classmethod
    def create(cls, display_name):
        endpoint = cls(display_name=display_name)
        endpoint.resource_name = f"projects/test/locations/test/endpoints/{display_name}"
        endpoint.display_name = display_name
        endpoint.name = endpoint.resource_name
        cls.registry[endpoint.resource_name] = endpoint
        return endpoint

    @classmethod
    def list(cls):
        return list(cls.registry.values())


def init(project=None, location=None):
    DummyEndpoint.init_args = {"project": project, "location": location}


ai_module.init = init
ai_module.Endpoint = DummyEndpoint
ai_module.Model = DummyModel
sys.modules["google.cloud.aiplatform"] = ai_module

# Export commonly used names for convenience in tests
Conflict = Conflict
GoogleCloudError = GoogleCloudError
NotFound = NotFound


@pytest.fixture(autouse=True)
def reset_ai_endpoint_registry():
    """Ensure the dummy AI Platform registry is clean for each test."""
    DummyEndpoint.registry.clear()
    DummyEndpoint.init_args.clear()
    yield
