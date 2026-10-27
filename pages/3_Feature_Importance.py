
import streamlit as st
import pandas as pd
import plotly.express as px

from src.utils.storage import list_saved_models, load_model

st.set_page_config(
    page_title="Feature Importance | ModelForge",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Feature Importance Studio")
st.caption(
    "Discover which input features contribute most to a trained model's predictions."
)

st.divider()

saved_models = list_saved_models()

if not saved_models:
    st.warning("No saved models found. Train a model in Dataset Studio first.")
    st.stop()

selected_model = st.selectbox(
    "Select a trained model",
    saved_models,
)

try:
    model = load_model(selected_model)

    # The project saves preprocessing and the estimator in a Pipeline.
    if not hasattr(model, "named_steps"):
        st.error("This page currently supports saved preprocessing pipelines only.")
        st.stop()

    estimator = model.named_steps.get("model")
    preprocessor = model.named_steps.get("preprocessor")

    if estimator is None or preprocessor is None:
        st.error("The selected model does not contain the expected pipeline steps.")
        st.stop()

    # Extract importance values from supported estimators.
    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
        importance_type = "Feature importance"

    elif hasattr(estimator, "coef_"):
        # For linear models, use absolute coefficient magnitude.
        # For multiclass classification, average across classes.
        coefficients = estimator.coef_

        if coefficients.ndim == 2:
            importance_values = abs(coefficients).mean(axis=0)
        else:
            importance_values = abs(coefficients)

        importance_type = "Absolute coefficient magnitude"

    else:
        st.info(
            "This estimator does not expose built-in feature importance "
            "or coefficients. Try Random Forest or Gradient Boosting."
        )
        st.stop()

    # Get names after preprocessing, including encoded categorical features.
    try:
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        feature_names = [
            f"Feature {i + 1}" for i in range(len(importance_values))
        ]

    if len(feature_names) != len(importance_values):
        feature_names = [
            f"Feature {i + 1}" for i in range(len(importance_values))
        ]

    results = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values,
    })

    # Make the labels easier to read.
    results["Feature"] = (
        results["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    results = results.sort_values("Importance", ascending=False)

    st.subheader("Importance Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected model", selected_model)
    col2.metric("Features analyzed", len(results))
    col3.metric("Top feature", results.iloc[0]["Feature"])

    st.divider()

    st.subheader("Feature Ranking")

    top_n = st.slider(
        "Number of features to display",
        min_value=1,
        max_value=len(results),
        value=min(10, len(results)),
    )

    chart_data = results.head(top_n).sort_values(
        "Importance", ascending=True
    )

    fig = px.bar(
        chart_data,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Most influential features",
        labels={
            "Importance": importance_type,
            "Feature": "Input feature",
        },
        text_auto=".3f",
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View complete feature ranking"):
        st.dataframe(
            results.reset_index(drop=True),
            use_container_width=True,
        )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download feature ranking as CSV",
        data=csv_data,
        file_name="modelforge_feature_importance.csv",
        mime="text/csv",
    )

    st.caption(
        "Interpretation note: tree-based importance and coefficient magnitude "
        "measure different things. Neither proves that a feature causes an outcome."
    )

except Exception as exc:
    st.error(f"Could not analyze this model: {exc}")
