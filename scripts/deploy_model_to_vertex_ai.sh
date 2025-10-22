set -e # Exit immediately if a command exits with a non-zero status
echo "Check if endpoint is deployed..."
python scripts/model/deploy_model_to_vertex.py
echo "Running inference on sample JSON payload..."
python scripts/predict/run_inference.py --json sample_data/payload.json
echo "Inference completed."
