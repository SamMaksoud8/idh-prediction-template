from google.cloud import aiplatform
from idh.model.endpoint import get_endpoint_id_from_config
from idh.config import config


def prepare_payload_for_inference(payload: dict) -> tuple[list, dict]:
    """
    Prepare payload for Vertex AI inference.

    The expected payload structure is:
        {
            "instances": [...],    # optional list of instances
            "parameters": {...}    # optional dict of parameters
        }

    Args:
        payload: A mapping that may contain "instances" and "parameters".

    Returns:
        A tuple (instances, parameters) where instances is a list (defaults to [])
        and parameters is a dict (defaults to {}).
    """
    if payload is None:
        payload = {}
    instances = payload.get("instances", [])
    parameters = payload.get("parameters", {})
    return instances, parameters


def predict(
    project_id: str,
    region: str,
    endpoint_id: str,
    instances: list,
    parameters: dict = None,
) -> object:
    """
    Perform a prediction request against a Vertex AI endpoint.

    Args:
        project_id: GCP project id.
        region: GCP region (location).
        endpoint_id: Vertex AI endpoint identifier.
        instances: A list of instances to be sent for prediction.
        parameters: Optional dictionary of prediction parameters.

    Returns:
        The raw response object returned by the Vertex AI endpoint.predict call.

    Raises:
        Any exception propagated from the Google Cloud AI Platform client when the
        prediction request fails.
    """
    aiplatform.init(project=project_id, location=region)
    endpoint = aiplatform.Endpoint(
        endpoint_name=f"projects/{project_id}/locations/{region}/endpoints/{endpoint_id}"
    )
    response = endpoint.predict(instances=instances, parameters=parameters)
    return response


def predict_from_config(instances: list, parameters: dict = None) -> object:
    """
    Perform a prediction using project/region/endpoint values from the config.

    Args:
        instances: A list of instances to send for prediction.
        parameters: Optional dict of prediction parameters.

    Returns:
        The raw response object returned by predict().

    Raises:
        Any exception raised by get_endpoint_id_from_config or predict is propagated.
    """
    project_id = config.project_name
    region = config.region
    endpoint_id = get_endpoint_id_from_config()
    return predict(project_id, region, endpoint_id, instances, parameters)
