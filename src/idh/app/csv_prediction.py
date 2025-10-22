"""Gradio interface for predicting IDH risk from uploaded CSV files."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import argparse
import os

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import idh.data.session as session_data
import idh.model.predict as predict
from idh.config import config
from idh.model.endpoint import get_endpoint_id_from_config

PROJECT_ID = config.project_name
REGION = config.region
ENDPOINT_ID = get_endpoint_id_from_config()
print(f"Using model endpoint {ENDPOINT_ID}.")


def resolve(preds: Sequence[Mapping[str, Any]]) -> bool:
    """Return ``True`` if any prediction in ``preds`` indicates high IDH risk."""
    for pred in preds:
        if pred.get("predicted_label") == "1":
            return True
    return False


def create_sbp_plot(df: pd.DataFrame) -> go.Figure | None:
    """Generate an interactive Plotly chart of SBP over time."""
    if "datatime" not in df.columns or "sbp" not in df.columns:
        return None  # Return nothing if columns are missing

    fig = px.line(
        df,
        x="datatime",
        y="sbp",
        markers=True,  # Adds dots to the data points
        title="Systolic Blood Pressure (SBP) Over Time",
        labels={"datatime": "Time", "sbp": "SBP (mmHg)"},
    )

    fig.update_traces(hovertemplate="<b>Time</b>: %{x}<br><b>SBP</b>: %{y} mmHg")
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="SBP (mmHg)",
        template="plotly_white",  # Use a clean theme
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig


def predict_from_csv(uploaded_file: Any | None) -> tuple[str, go.Figure | None] | str:
    """Load a CSV from Gradio, invoke the model, and return status plus plot."""
    if uploaded_file is None:
        return "Please upload a CSV file to begin."

    filepath = uploaded_file.name
    filename = os.path.basename(filepath)

    if not filepath.lower().endswith(".csv"):
        return f"Error: Invalid file type for '{filename}'. Please upload a .csv file."

    try:
        print(f"Processing file: {filename}")

        df = session_data.load_from_csv(filepath)
        payload = session_data.session_data_to_vertex_json(df)
        print(f"Successfully loaded {filename}.")

        print("--- Running Model Inference ---")
        instances, parameters = predict.prepare_payload_for_inference(payload)

        sbp_chart = create_sbp_plot(df)

        result = predict.predict(PROJECT_ID, REGION, ENDPOINT_ID, instances, parameters)

        if resolve(result.predictions):
            message = "⚠️ High IDH Risk"
        else:
            message = "✅ Low IDH Risk"

        if sbp_chart:
            return message, sbp_chart
        return message, None

    except Exception as exc:  # pragma: no cover - defensive logging for UI handler
        error_message = f"❌ An error occurred while processing {filename}:\n\n{exc}"
        print(error_message)
        return error_message


with gr.Blocks(theme=gr.themes.Soft()) as app_interface:
    gr.Markdown("# IDH Event Predictor with SBP Visualization")
    gr.Markdown(
        "Upload a CSV file to predict the likelihood of an IDH event. This model defines an IDH event as an **sbp<90mmHg**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Drag & Drop CSV File Here", file_types=[".csv"])
            predict_button = gr.Button("Run Prediction", variant="primary")

        with gr.Column(scale=2):
            result_label = gr.Label(label="Prediction Result")
            sbp_plot = gr.Plot(label="SBP Over Time")

    predict_button.click(
        fn=predict_from_csv,
        inputs=file_input,
        outputs=[result_label, sbp_plot],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IDH prediction Gradio app.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="If set, create a publicly shareable Gradio link.",
    )
    args = parser.parse_args()

    app_interface.launch(
        server_name="0.0.0.0",
        share=args.share,
        server_port=int(os.getenv("PORT", "8080")),
    )
