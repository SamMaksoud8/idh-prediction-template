from idh.config import config
import argparse
import idh.gcp.bigquery.query as query
from google.cloud import bigquery
import idh.gcp.bigquery.tables as bq_tables


def check_data_tables_ready(client):
    print("Checking if data tables are ready...")
    bq_tables.check_table_ready(client, table_id=bq_tables.patient_demographics_table_id())
    bq_tables.check_table_ready(client, table_id=bq_tables.registration_data_table_id())
    bq_tables.check_table_ready(client, table_id=bq_tables.real_time_machine_data_table_id())


def check_features_table_ready(client):
    print("Checking if features table is ready...")
    bq_tables.check_table_ready(client, table_id=bq_tables.features_table_id())


def run_model_training():
    client = bigquery.Client(project=config.project_name)
    check_data_tables_ready(client)
    print("Extracting and engineering features in BigQuery...")
    query.sessionize_machine_data_from_config()
    query.features_engineering_from_config()
    print("Feature engineering completed.")
    check_features_table_ready(client)
    print("Starting model training...")
    query.run_model_training_from_config()
    print("Model training completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--skip-feature-extraction",
        "-s",
        action="store_true",
        help="Skip extracting and engineering features in BigQuery",
    )
    args, _ = parser.parse_known_args()

    if args.skip_feature_extraction:

        def _skip_sessionize():
            print("Skipping sessionize (feature extraction) as requested.")

        def _skip_features_engineering():
            print("Skipping features engineering as requested.")

        query.sessionize_machine_data_from_config = _skip_sessionize
        query.features_engineering_from_config = _skip_features_engineering
    run_model_training()
