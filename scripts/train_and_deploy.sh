#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status,
set -e

echo "Starting data setup..."
python scripts/model/setup_data.py
echo "Data setup completed."

echo "Starting model training..."
python scripts/model/run_model_training.py
echo "Model training completed."

echo "Starting model evaluation..."
python scripts/model/run_model_evaluation.py
echo "Model evaluation completed."

echo "Deploying model to Vertex AI..."
python scripts/model/deploy_model_to_vertex.py
echo "Model deployment completed."

echo "Running inference on sample JSON payload..."
python scripts/predict/run_inference.py --json sample_data/payload.json
echo "Inference completed."