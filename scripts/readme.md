## Common workflows

### 1. Provision data infrastructure
The setup script creates the Cloud Storage bucket, uploads source CSVs, converts them to Parquet,
and creates BigQuery datasets/tables.
```bash
python scripts/model/setup_data.py
```
Ensure your active Google Cloud credentials point to the project specified in `config.yaml`.

### 2. Feature engineering and model training
Trigger the feature engineering queries (sessionisation + feature creation) and BigQuery ML model
training:
```bash
python scripts/model/run_model_training.py
```
Use `--skip-feature-extraction` if you only want to retrain on existing engineered features.

### 3. Model evaluation
Review the performance of the trained model using:
```bash
python scripts/model/run_model_evaluation.py
```

### 4. Deploy to Vertex AI
Deploy the BigQuery ML model as an endpoint that the application and CLI tools can query:
```bash
python scripts/model/deploy_model_to_vertex.py
```

### 5. Run online predictions from the command line
Invoke the deployed endpoint using either a JSON payload or a CSV session:
```bash
# Using the provided JSON payload
python scripts/predict/run_inference.py --json sample_data/payload.json

# Converting a CSV session to the expected payload on-the-fly
python scripts/predict/run_inference.py --csv sample_data/idh_session.csv
```

## Testing the deployment

Use the helper script to confirm that the endpoint is ready and accepts the sample payload:
```bash
bash scripts/test_prediction.sh
```
This script reuses the deployment helper to ensure a model is available and then executes the
inference CLI against `sample_data/payload.json`.

## Additional utilities
### Generate sample CSVs for the Gradio app
Use the helper to fetch a dialysis session from BigQuery and save it locally as a test CSV for the
Gradio interface so you can exercise any session id end-to-end:
```bash
python scripts/predict/create_sample_csv.py --project-id <your-gcp-project> --session-id <session-id>
```
The script stores the CSV in the working directory by default, making it easy to build a library of
test sessions for demos or troubleshooting.

### Create a Vertex AI prediction payload
Build a JSON payload for a specific session id—either by pulling the source data from BigQuery or by
converting an existing CSV—so you can exercise the deployed Vertex AI endpoint directly:
```bash
python scripts/predict/create_vertex_json.py --project-id <your-gcp-project> --session-id <session-id> --save-path payload.json
```
You can also provide a `--csv` path instead of `--project-id/--session-id` if you already have a
local file and simply need to format it for the prediction CLI.

