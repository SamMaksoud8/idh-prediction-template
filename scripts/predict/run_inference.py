"""
Run inference using the project's configured Vertex AI model.
This module exposes a convenience function run_inference(instances, parameters)
which delegates to idh.model.predict.predict_from_config, and also provides a
simple command-line interface for loading an inference payload from a local
JSON file or converting a CSV to the required payload format.
Command-line usage:
    python run_inference.py --json /path/to/payload.json
    python run_inference.py --csv  /path/to/data.csv
JSON payload:
    The JSON file should contain the payload accepted by
    idh.model.predict.prepare_payload_for_inference — commonly a dict that
    includes 'instances' (a list of instance dicts) and optionally
    'parameters' (a dict of model-specific prediction parameters).
CSV payload:
    When a CSV path is supplied, the file is converted to the expected JSON
    payload via idh.data.session.csv_to_vertex_json before being prepared for
    inference.
Function:
    run_inference(instances: List[Dict[str, Any]], parameters: Dict[str, Any]) -> Any
    - instances: a list of instance dictionaries to send to the model.
    - parameters: prediction parameters (model-specific).
    - returns: the raw response from predict.predict_from_config, typically
      containing fields such as deployed_model_id and predictions.
Behavior:
    - The CLI requires exactly one of --json or --csv to be provided.
    - The script prints the deployed model id and enumerates the predictions
      returned by the model to stdout.
"""

import json
import argparse
import idh.model.predict as predict
import idh.data.session as session_data
from idh.config import config
from typing import Any, Dict, List


def run_inference(instances: List[Dict[str, Any]], parameters: Dict[str, Any]) -> Any:
    """
    Run prediction using the configured Vertex AI model.

    Args:
        instances: A list of instance dicts to send for inference.
        parameters: A dict of prediction parameters (model-specific).

    Returns:
        The raw response from predict.predict_from_config (typically contains
        deployed_model_id and predictions).
    """
    return predict.predict_from_config(instances, parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple inference on a Vertex AI endpoint.")
    parser.add_argument(
        "--json",
        type=str,
        help="Path to the local JSON file containing the inference payload.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the local JSON file containing the inference payload.",
    )

    args = parser.parse_args()

    if args.json:
        # Load the JSON payload file
        with open(args.json, "r") as f:
            payload = json.load(f)
    elif args.csv:
        payload = session_data.csv_to_vertex_json(args.csv)
    else:
        raise ValueError("Cannot proceed because either a json or csv path must be provided")

    instances, parameters = predict.prepare_payload_for_inference(payload)
    result = run_inference(instances, parameters)
    print("Deployed model ID:", result.deployed_model_id)
    print("Predictions:")
    for i, pred in enumerate(result.predictions):
        print(f"[{i}] {pred}")
