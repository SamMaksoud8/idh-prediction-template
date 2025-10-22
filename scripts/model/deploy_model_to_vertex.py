"""
Deploy the trained model to Google Cloud Vertex AI and ensure an endpoint exists.
This module provides a single convenience function, deploy_model_to_vertex(), which:
- Looks up an existing Vertex AI endpoint by the display name specified in the project
    configuration.
- If no endpoint is found, creates a new Vertex AI endpoint.
- Deploys the configured model to the retrieved or newly created endpoint using the
    machine type and replica settings defined in the configuration.
Functions:
        deploy_model_to_vertex() -> None
                - Uses values from idh.config.config:
                        - config.project_name: GCP project ID
                        - config.region: Vertex AI region
                        - config.model.endpoint_name: endpoint display name
                        - config.model.name: model identifier/name to deploy
                        - config.model.machine_type: machine type for serving
                        - config.model.min_replicas: minimum number of replicas
                        - config.model.max_replicas: maximum number of replicas
                - Relies on helper functions from idh.model.endpoint:
                        - get_endpoint(project_id, location, endpoint_display_name)
                        - create_vertex_endpoint(project_id, region, endpoint_name)
                        - deploy_vertex_model(project_id, region, model_name, endpoint, machine_type, min_replicas, max_replicas)
                - Performs no return; side effects include creating/retrieving an endpoint and deploying a model.
Operational notes:
        - Requires valid GCP credentials and Vertex AI API access for the target project.
        - Errors from the underlying helper functions or the Vertex AI service will propagate,
            so callers should run this in an environment prepared for handling exceptions/logging.
        - Designed to be executable as a script for simple CI/CD or manual deployments.
"""

from idh.config import config
from idh.model.endpoint import (
    create_vertex_endpoint,
    deploy_vertex_model,
    get_endpoint,
    check_endpoint_deployed,
)


def deploy_model_to_vertex() -> None:
    """Deploy the trained model to Vertex AI Model Registry."""
    endpoint = get_endpoint(
        project_id=config.project_name,
        location=config.region,
        endpoint_display_name=config.model.endpoint_name,
    )
    if endpoint is None:
        endpoint = create_vertex_endpoint(
            project_id=config.project_name,
            region=config.region,
            endpoint_name=config.model.endpoint_name,
        )
    if check_endpoint_deployed(endpoint):
        print(f"Model already deployed to endpoint '{endpoint.display_name}'. Skipping deployment.")
        return
    deploy_vertex_model(
        project_id=config.project_name,
        region=config.region,
        model_name=config.model.name,
        endpoint=endpoint,
        machine_type=config.model.machine_type,
        min_replicas=config.model.min_replicas,
        max_replicas=config.model.max_replicas,
    )


if __name__ == "__main__":
    deploy_model_to_vertex()
