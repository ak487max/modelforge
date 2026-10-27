
import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.storage import list_saved_models
from src.utils.experiment_tracker import get_experiment_history

st.set_page_config(
    page_title="Model Evaluation | ModelForge",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation Studio")
st.caption("Understand, evaluate, and compare your machine learning experiments.")

st.divider()

# Load saved models and experiment history
saved_models = list_saved_models()
history = get_experiment_history()

if not history:
    st.info(
        "No experiments found yet. Train a model in Dataset Studio "
        "to see its evaluation metrics here."
    )
    st.stop()

df = pd.DataFrame(history)

if df.empty:
    st.warning("Your experiment history is empty.")
    st.stop()

# Select experiment
if "model_name" not in df.columns:
    st.error("Experiment history does not contain model names.")
    st.stop()

df["experiment_label"] = (
    df["model_name"].astype(str)
    + " — "
    + df.index.astype(str)
)

selected_label = st.selectbox(
    "Select an experiment",
    df["experiment_label"].tolist(),
)

selected_index = df.index[
    df["experiment_label"] == selected_label
][0]

experiment = df.loc[selected_index]

st.subheader("Experiment Details")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Model",
    str(experiment.get("model_name", "Unknown")),
)

col2.metric(
    "Task",
    str(experiment.get("task", "Unknown")).title(),
)

col3.metric(
    "Dataset",
    str(experiment.get("dataset_name", "Unknown")),
)

st.divider()

# Display available evaluation metrics
metrics = experiment.get("metrics", {})

if isinstance(metrics, dict):
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
else:
    numeric_metrics = {}

if numeric_metrics:
    st.subheader("Performance Metrics")

    metric_columns = st.columns(min(len(numeric_metrics), 4))

    for i, (name, value) in enumerate(numeric_metrics.items()):
        with metric_columns[i % len(metric_columns)]:
            # Scores are displayed as percentages when between 0 and 1.
            if 0 <= value <= 1 and name in (
                "accuracy", "precision", "recall", "f1_score", "r2_score"
            ):
                st.metric(name.replace("_", " ").title(), f"{value:.2%}")
            else:
                st.metric(name.replace("_", " ").title(), f"{value:.4f}")
else:
    st.warning("No numeric evaluation metrics were found for this experiment.")

# Show a chart when numeric metrics exist
if numeric_metrics:
    st.subheader("Metric Comparison")

    chart_data = pd.DataFrame({
        "Metric": list(numeric_metrics.keys()),
        "Value": list(numeric_metrics.values()),
    })

    fig = px.bar(
        chart_data,
        x="Metric",
        y="Value",
        title="Metrics for the Selected Experiment",
        text_auto=".3f",
    )

    st.plotly_chart(fig, use_container_width=True)

with st.expander("View all experiment records"):
    st.dataframe(df.drop(columns=["experiment_label"]), use_container_width=True)

st.caption(
    f"Saved models available: {len(saved_models)}"
)

