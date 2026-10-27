
import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.experiment_tracker import get_experiment_history

st.set_page_config(
    page_title="Experiment Comparison | ModelForge",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 ModelForge Leaderboard")
st.caption("Rank and compare your machine learning experiments.")

history = get_experiment_history()

if not history:
    st.info("Train a model in Dataset Studio to get started.")
    st.stop()

records = []

for index, exp in enumerate(history):
    metrics = exp.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    record = {
        "Experiment": f"{exp.get('model_name', 'Unknown')} #{index + 1}",
        "Model": exp.get("model_name", "Unknown"),
        "Task": str(exp.get("task", metrics.get("task", "Unknown"))).lower(),
        "Dataset": exp.get("dataset_name", "Unknown"),
        "Target": exp.get("target_column", metrics.get("target_column", "Unknown")),
    }

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            record[key] = value

    records.append(record)

df = pd.DataFrame(records)

if df.empty:
    st.warning("No experiment records available.")
    st.stop()

# Select task first, so classification and regression are ranked separately.
tasks = sorted(df["Task"].dropna().unique().tolist())
selected_task = st.selectbox("Choose task", tasks)

task_df = df[df["Task"] == selected_task].copy()

if task_df.empty:
    st.info("No experiments found for this task.")
    st.stop()

# Higher is better for these metrics; lower is better for error metrics.
higher_is_better = [
    "accuracy", "precision", "recall", "f1_score", "roc_auc", "r2_score"
]
lower_is_better = ["mae", "mse", "rmse"]

available_metrics = [
    metric for metric in higher_is_better + lower_is_better
    if metric in task_df.columns and task_df[metric].notna().any()
]

if not available_metrics:
    st.warning("No supported numeric evaluation metrics found.")
    st.stop()

default_metric = (
    "accuracy" if selected_task == "classification" and "accuracy" in available_metrics
    else "r2_score" if selected_task == "regression" and "r2_score" in available_metrics
    else available_metrics[0]
)

selected_metric = st.selectbox(
    "Rank models by",
    available_metrics,
    index=available_metrics.index(default_metric),
)

task_df = task_df.dropna(subset=[selected_metric]).copy()

if task_df.empty:
    st.warning("No experiments contain this metric.")
    st.stop()

ascending = selected_metric in lower_is_better

# Keep the best experiment first.
leaderboard = task_df.sort_values(
    selected_metric,
    ascending=ascending,
).reset_index(drop=True)

leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

st.divider()
st.subheader("🏆 Leaderboard")

display_columns = [
    col for col in [
        "Rank", "Experiment", "Model", "Dataset", "Target",
        "accuracy", "precision", "recall", "f1_score",
        "mae", "rmse", "r2_score"
    ]
    if col in leaderboard.columns
]

st.dataframe(
    leaderboard[display_columns],
    use_container_width=True,
    hide_index=True,
)

winner = leaderboard.iloc[0]

st.success(
    f"Top experiment: {winner['Experiment']} — "
    f"{selected_metric} = {winner[selected_metric]:.4f}"
)

st.caption(
    "Rankings are based only on the selected metric. "
    "Compare experiments using the same dataset, target, task, "
    "and evaluation split for a fairer comparison."
)

st.divider()
st.subheader("📊 Performance Comparison")

chart_df = leaderboard.sort_values(selected_metric, ascending=True)

fig = px.bar(
    chart_df,
    x=selected_metric,
    y="Experiment",
    orientation="h",
    text_auto=".4f",
    title=f"Models ranked by {selected_metric.replace('_', ' ').title()}",
)

st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "Download leaderboard CSV",
    data=leaderboard.to_csv(index=False).encode("utf-8"),
    file_name="modelforge_leaderboard.csv",
    mime="text/csv",
)

st.caption("ModelForge • Experiment tracking and evaluation")
