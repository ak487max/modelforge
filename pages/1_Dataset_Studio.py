
import streamlit as st
import pandas as pd
from src.utils.experiment_tracker import save_experiment
from src.utils.data_loader import load_csv, get_dataset_summary, validate_target
from src.ml.train import train_model


st.set_page_config(
    page_title="Dataset Studio | ModelForge",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dataset Studio")
st.caption("Upload, inspect, and train machine learning models on your own data.")

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
    help="Choose a dataset with feature columns and a target column.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

try:
    data = load_csv(uploaded_file)
except (ValueError, TypeError) as exc:
    st.error(str(exc))
    st.stop()

summary = get_dataset_summary(data)

st.subheader("Dataset overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", summary["rows"])
col2.metric("Columns", summary["columns"])
col3.metric("Missing values", summary["missing_values"])
col4.metric("Duplicate rows", summary["duplicate_rows"])

st.subheader("Data preview")
st.dataframe(data.head(100), use_container_width=True)

with st.expander("Column details"):
    details = pd.DataFrame({
        "Column": data.columns,
        "Data type": [str(dtype) for dtype in data.dtypes],
        "Missing values": data.isna().sum().values,
        "Unique values": data.nunique().values,
    })
    st.dataframe(details, use_container_width=True)

st.divider()
st.subheader("Configure your experiment")

target_column = st.selectbox(
    "Select the target column to predict",
    options=data.columns.tolist(),
)

task = st.radio(
    "Machine learning task",
    options=["classification", "regression"],
    horizontal=True,
    help="Choose classification for categories and regression for numeric quantities.",
)

if task == "classification":
    model_options = [
        "Random Forest",
        "Gradient Boosting",
        "Logistic Regression",
    ]
else:
    model_options = [
        "Random Forest",
        "Gradient Boosting",
        "Linear Regression",
    ]

model_name = st.selectbox("Choose a model", model_options)

test_size = st.slider(
    "Test dataset percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
) / 100

if st.button("🚀 Train Model", type="primary", use_container_width=True):
    try:
        validate_target(data, target_column)

        with st.spinner("Training and evaluating your model..."):
            model, metrics = train_model(
                data=data,
                target_column=target_column,
                model_name=model_name,
                test_size=test_size,
                task=task,
            )

        st.session_state["last_training_metrics"] = metrics
        st.session_state["last_training_model"] = model
        st.session_state["last_training_target"] = target_column

        st.success("Training completed and model saved!")
        experiment_id = save_experiment(
            model_name=model_name,
            dataset_name=uploaded_file.name,
            task=task,
            metrics=metrics,
            target_column=target_column,
)

        st.success("Experiment saved to history!")
        st.caption(f"Experiment ID: {experiment_id}")

        st.subheader("Evaluation results")

        if task == "classification":
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            m2.metric("Precision", f"{metrics['precision']:.2%}")
            m3.metric("Recall", f"{metrics['recall']:.2%}")
            m4.metric("F1 score", f"{metrics['f1_score']:.2%}")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{metrics['mae']:.4f}")
            m2.metric("RMSE", f"{metrics['rmse']:.4f}")
            m3.metric("R²", f"{metrics['r2_score']:.4f}")

        st.write("**Saved model:**", metrics["model_path"])
        st.write("**Training samples:**", metrics["train_samples"])
        st.write("**Testing samples:**", metrics["test_samples"])

    except (ValueError, TypeError) as exc:
        st.error(f"Could not train model: {exc}")
    except Exception as exc:
        st.error(f"Unexpected training error: {exc}")
