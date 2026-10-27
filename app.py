
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

from src.ml.train import train_model
from src.ml.evaluate import evaluate_model
from src.ml.predict import predict_sample
from src.utils.storage import list_saved_models


# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="ModelForge | ML Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: #0B1020;
    }

    [data-testid="stSidebar"] {
        background: #11182A;
        border-right: 1px solid #252D42;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 0;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-top: 5px;
    }

    div[data-testid="stMetric"] {
        background: #151D31;
        border: 1px solid #293550;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        background: #151D31;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #293550;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# ⚡ ModelForge")
    st.caption("MACHINE LEARNING STUDIO")
    st.divider()

    st.markdown("### WORKSPACE")
    st.write("Dataset: Iris")
    st.write("Task: Classification")
    st.write("Execution: Local Python")

    st.divider()
    st.caption("NO DOCKER • NO API SERVER")


# ---------------- HEADER ----------------
st.markdown(
    '<p class="main-title">ML Studio Overview</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    'Train, evaluate and explore machine learning models.'
    '</p>',
    unsafe_allow_html=True,
)

st.write("")
st.divider()


# ---------------- LOAD SAVED METRICS ----------------
saved_models = list_saved_models()
selected_model = "Random Forest"

metrics = None

if "dashboard_metrics" not in st.session_state:
    try:
        metrics = evaluate_model(
            model_name=selected_model,
            target_column="target",
            task="classification",
        )
        st.session_state["dashboard_metrics"] = metrics
    except Exception as exc:
        st.error(f"Unable to evaluate the model: {exc}")

metrics = st.session_state.get("dashboard_metrics")


# ---------------- METRIC CARDS ----------------
st.markdown("### Workspace Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Models Saved",
        len(saved_models),
    )

with col2:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}" if metrics else "—",
    )

with col3:
    st.metric(
        "F1 Score",
        f"{metrics['f1_score']:.2%}" if metrics else "—",
    )

with col4:
    st.metric(
        "Test Samples",
        metrics["test_samples"] if metrics else "—",
    )

st.write("")
st.divider()


# ---------------- MODEL STATUS ----------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### Model Status")

    if saved_models:
        st.success("Saved model detected")

        for model_name in saved_models:
            st.write(f"• `{model_name}`")
    else:
        st.warning("No saved model found. Train a model to get started.")

    if metrics:
        st.write(f"**Training samples:** {metrics['train_samples']}")
        st.write(f"**Testing samples:** {metrics['test_samples']}")
        st.write(f"**Evaluation model:** {metrics['model_name']}")

    if st.button("🔄 Retrain Random Forest", use_container_width=True):
        with st.spinner("Training model and calculating metrics..."):
            try:
                model, _ = train_model(
                    model_name="Random Forest",
                    target_column="target",
                    task="classification",
)
                updated_metrics = evaluate_model("Random Forest")
                st.session_state["dashboard_metrics"] = updated_metrics
                st.success("Model trained and metrics refreshed!")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")


# ---------------- ACCURACY VISUALIZATION ----------------
with right_col:
    st.markdown("### Performance Snapshot")

    if metrics:
        chart = go.Figure(
            go.Bar(
                x=["Accuracy", "Precision", "Recall", "F1 Score"],
                y=[
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1_score"],
                ],
                text=[
                    f"{metrics['accuracy']:.1%}",
                    f"{metrics['precision']:.1%}",
                    f"{metrics['recall']:.1%}",
                    f"{metrics['f1_score']:.1%}",
                ],
                textposition="auto",
            )
        )

        chart.update_layout(
            height=330,
            yaxis=dict(
                title="Score",
                range=[0, 1],
                tickformat=".0%",
            ),
            paper_bgcolor="#151D31",
            plot_bgcolor="#151D31",
            font=dict(color="#F1F5F9"),
            margin=dict(l=20, r=20, t=25, b=20),
        )

        st.plotly_chart(chart, use_container_width=True)


# ---------------- PREDICTION STUDIO ----------------
st.divider()
st.markdown("### 🧪 Quick Prediction Studio")

st.write(
    "Enter four flower measurements to predict the Iris species "
    "using your saved Random Forest model."
)

if "random_forest" not in saved_models:
    st.info("Train and save the Random Forest model before predicting.")
else:
    with st.form("prediction_form"):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            sepal_length = st.number_input(
                "Sepal length (cm)", min_value=0.1,
                max_value=10.0, value=5.1, step=0.1,
            )

        with c2:
            sepal_width = st.number_input(
                "Sepal width (cm)", min_value=0.1,
                max_value=10.0, value=3.5, step=0.1,
            )

        with c3:
            petal_length = st.number_input(
                "Petal length (cm)", min_value=0.1,
                max_value=10.0, value=1.4, step=0.1,
            )

        with c4:
            petal_width = st.number_input(
                "Petal width (cm)", min_value=0.1,
                max_value=10.0, value=0.2, step=0.1,
            )

        submitted = st.form_submit_button(
            "Predict Species",
            use_container_width=True,
        )

    if submitted:
        try:
            result = predict_sample(
                sepal_length,
                sepal_width,
                petal_length,
                petal_width,
            )

            st.success(f"Predicted species: **{result['species'].title()}**")

            if result["probabilities"]:
                st.markdown("#### Class probabilities")

                for species, probability in result["probabilities"].items():
                    st.progress(
                        probability,
                        text=f"{species.title()}: {probability:.1%}",
                    )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ---------------- FOOTER ----------------
st.divider()
st.caption("ModelForge • Local-first machine learning platform")
