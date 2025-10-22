from idh.config import config
import numbers
import idh.gcp.bigquery.query as query


def _format_and_print_results(results):
    """Format the results dataframe and print it."""
    df = results.copy()
    # round numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)
    # print column: value pairs if single-row
    print()
    print("Model Evaluation Results:")
    if len(df) == 1:
        row = df.iloc[0]
        for col, val in row.items():
            if isinstance(val, numbers.Number) and not isinstance(val, bool):
                print(f"{col}: {val:.4f}")
            else:
                print(f"{col}: {val}")
    else:
        print(df.to_string())


def run_model_evaluation():
    print("Starting model evaluation...")
    results = query.run_model_evaluation_from_config()
    print("Model evaluation completed.")
    df = results.to_dataframe()
    _format_and_print_results(df)
    df.to_csv(
        f"model_evaluation_results_{config.model.name}.csv", index=False
    )


if __name__ == "__main__":
    run_model_evaluation()
