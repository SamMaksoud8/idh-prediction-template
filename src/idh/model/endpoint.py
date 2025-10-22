from google.cloud import aiplatform
from idh.config import config


def check_endpoint_deployed(endpoint: aiplatform.Endpoint) -> bool:
    """Return True if the endpoint has deployed models, else False."""
    # Reload full endpoint resource
    full_endpoint = aiplatform.Endpoint(endpoint.resource_name).gca_resource

    # Extract deployed models (if any)
    deployed_models = getattr(full_endpoint, "deployed_models", [])

    return bool(deployed_models)


def create_vertex_endpoint(project_id: str, region: str, endpoint_name: str) -> aiplatform.Endpoint:
    # Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=region)

    # Create the Endpoint
    print(f"Creating endpoint '{endpoint_name}' in project '{project_id}' and region '{region}'...")
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    print(f"Endpoint created with resource name: {endpoint.resource_name}")

    return endpoint


def deploy_vertex_model(
    project_id: str,
    region: str,
    model_name: str,
    endpoint: aiplatform.Endpoint,
    env_file_path: str = ".env",  # Default path for the .env file
    machine_type: str = "n1-standard-2",
    min_replicas: int = 1,
    max_replicas: int = 2,
):
    # 1. Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=region)

    # ... (Model retrieval and endpoint creation logic is the same) ...
    try:
        model = aiplatform.Model(model_name=model_name)
    except Exception as e:
        print(f"Error retrieving model '{model_name}': {e}")
        return

    # 2. Deploy the Model
    print(f"Deploying model '{model.display_name}' to endpoint '{endpoint.display_name}'...")
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{model_name}-deployed",
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_split={"0": 100},
    )

    # 3. Extract the resource name (the full path)
    # This is the resource identifier used by the API:
    # projects/{project_id}/locations/{region}/endpoints/{endpoint_id}
    endpoint_resource_name = endpoint.resource_name

    # 4. Save the endpoint ID to the .env file
    try:
        with open(env_file_path, "a") as f:
            f.write(f"MODEL_ENDPOINT={endpoint_resource_name.split('/')[-1]}\n")
        print(f"Successfully saved endpoint resource name to {env_file_path} as MODEL_ENDPOINT.")
    except Exception as e:
        print(f"Error writing to .env file: {e}")

    return endpoint


def deploy_model_from_config(endpoint, env_file_path: str = ".env"):
    """Deploys a model to Vertex AI using configuration from the global config object.
    Args:
        env_file_path (str): Path to the .env file where the endpoint ID will be saved.
            Defaults to ".env".
    Returns:
        aiplatform.Endpoint: The deployed Vertex AI Endpoint object.
    """
    return deploy_vertex_model(
        project_id=config.project_name,
        region=config.region,
        model_name=config.model.name,
        endpoint=endpoint,
        env_file_path=env_file_path,
        machine_type=config.model.machine_type,
        min_replicas=config.model.min_replicas,
        max_replicas=config.model.max_replicas,
    )


def get_endpoint(project_id: str, location: str, endpoint_display_name: str) -> str | None:
    # Initialize the client
    aiplatform.init(project=project_id, location=location)

    # List all endpoints
    endpoints: list[aiplatform.Endpoint] = aiplatform.Endpoint.list()

    # Loop through the endpoints to find the one with the matching display name
    for endpoint in endpoints:
        if endpoint.display_name == endpoint_display_name:
            print(f"Found endpoint: '{endpoint.display_name}' with resource name: {endpoint.name}")
            return endpoint  # Full resource name (e.g. projects/.../endpoints/123)

    print(f"Endpoint with display name '{endpoint_display_name}' not found.")
    return None


def get_endpoint_id(project_id: str, location: str, endpoint_display_name: str) -> str | None:
    """Find the full Vertex AI Endpoint resource name for a given display name.
    Initializes the Vertex AI SDK for the provided project and location, lists
    all endpoints in that location, and returns the full resource name of the
    first endpoint whose display_name matches endpoint_display_name.
    Args:
        project_id (str): Google Cloud project ID used to initialize the Vertex AI client.
        location (str): Google Cloud location/region (for example, "us-central1").
        endpoint_display_name (str): The display name of the endpoint to search for.
    Returns:
        str | None: The full endpoint resource name (for example,
            "projects/{project}/locations/{location}/endpoints/{endpoint_id}") if a
            matching endpoint is found; otherwise None.
    Raises:
        Exception: Propagates exceptions raised by the Vertex AI SDK (for example,
            failures during client initialization or when listing endpoints). Callers
            may catch more specific SDK exceptions as needed.
    Example:
        >>> get_endpoint_id("my-project", "us-central1", "my-endpoint")
        "projects/my-project/locations/us-central1/endpoints/1234567890"
    """
    endpoint = get_endpoint(project_id, location, endpoint_display_name)
    if endpoint:
        return endpoint.name
    return None


def get_endpoint_id_from_config() -> str | None:
    """Resolve the AI Platform endpoint ID from the current configuration.

    If `config.model.endpoint` is set, that value is returned directly.
    Otherwise this function attempts to construct an endpoint identifier by calling
    `get_endpoint_id()` with the configured project, region, and model endpoint
    display name:

        project_id = config.project_name
        location = config.region
        endpoint_display_name = config.model.endpoint_name

    Returns:
        str | None: The resolved endpoint ID, or None if an endpoint could not be
        determined (for example, if required configuration values are missing or
        `get_endpoint_id()` returns None).
    """
    if config.model.endpoint:
        return config.model.endpoint
    project_id: str = config.project_name
    location: str = config.region
    endpoint_display_name: str = config.model.endpoint_name
    return get_endpoint_id(project_id, location, endpoint_display_name)


if __name__ == "__main__":
    print(get_endpoint_id_from_config())
