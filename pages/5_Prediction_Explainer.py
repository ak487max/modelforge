
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.utils.storage import list_saved_models, load_model

st.set_page_config(
    page_title="Prediction Explainer | ModelForge",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Prediction Explainer")
st.caption("Make predictions and inspect how input features affect the result.")
st.divider()

saved_models = list_saved_models()

if not saved_models:
    st.warning("No saved models found. Train a model in Dataset Studio first.")
    st.stop()

selected_model = st.selectbox("Choose a saved model", saved_models)
model = load_model(selected_model)

if not hasattr(model, "named_steps"):
    st.error("This page requires a saved preprocessing pipeline.")
    st.stop()

preprocessor = model.named_steps.get("preprocessor")
estimator = model.named_steps.get("model")

if preprocessor is None or estimator is None:
    st.error("The saved model is missing the expected pipeline steps.")
    st.stop()

if hasattr(preprocessor, "feature_names_in_"):
    feature_names = list(preprocessor.feature_names_in_)
else:
    st.error("Could not determine the original input columns.")
    st.stop()

st.subheader("Enter Input Values")
st.write("Provide values for one sample to generate a prediction.")

input_values = {}

with st.form("prediction_form"):
    columns = st.columns(2)

    for index, feature in enumerate(feature_names):
        with columns[index % 2]:
            input_values[feature] = st.text_input(
                feature.replace("_", " ").title(),
                placeholder="Enter a value",
            )

    submitted = st.form_submit_button("Generate Prediction")

if submitted:
    try:
        row = {}

        for feature in feature_names:
            raw_value = input_values[feature].strip()

            if not raw_value:
                raise ValueError(f"Please enter a value for {feature}.")

            try:
                row[feature] = float(raw_value)
            except ValueError:
                row[feature] = raw_value

        sample = pd.DataFrame([row], columns=feature_names)

        # Generate the prediction.
        prediction = model.predict(sample)[0]

        # Map labels only for the known Iris model.
        iris_labels = {
            0: "Setosa",
            1: "Versicolor",
            2: "Virginica",
        }

        is_iris_model = (
            selected_model == "classification_random_forest"
            and feature_names == [
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ]
        )

        if is_iris_model:
            prediction_display = iris_labels.get(
                prediction, str(prediction)
            )
        else:
            prediction_display = str(prediction)

        st.divider()
        st.subheader("Prediction Result")
        st.success(f"Predicted output: {prediction_display}")

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(sample)[0]
                classes = estimator.classes_

                probability_labels = [
                    iris_labels.get(value, str(value))
                    if is_iris_model
                    else str(value)
                    for value in classes
                ]

                probability_df = pd.DataFrame({
                    "Class": probability_labels,
                    "Probability": probabilities,
                }).sort_values("Probability", ascending=False)

                st.subheader("Class Probabilities")

                probability_fig = px.bar(
                    probability_df,
                    x="Class",
                    y="Probability",
                    text=probability_df["Probability"].map(
                        lambda value: f"{value:.1%}"
                    ),
                    title="Predicted probability by class",
                )

                probability_fig.update_yaxes(
                    tickformat=".0%",
                    range=[0, 1],
                )

                st.plotly_chart(
                    probability_fig,
                    use_container_width=True,
                )

            except (AttributeError, ValueError, TypeError):
                st.info("Class probabilities are unavailable for this model.")

        st.subheader("What Influences This Model's Predictions?")

        if hasattr(estimator, "feature_importances_"):
            importance = estimator.feature_importances_
            explanation_method = "Global tree-based feature importance"

        elif hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_)

            if coefficients.ndim == 2:
                importance = np.mean(np.abs(coefficients), axis=0)
            else:
                importance = np.abs(coefficients)

            explanation_method = "Absolute coefficient magnitude"

        else:
            importance = None
            explanation_method = None

        if importance is not None:
            try:
                transformed_names = preprocessor.get_feature_names_out()
            except (AttributeError, ValueError):
                transformed_names = [
                    f"Feature {i + 1}"
                    for i in range(len(importance))
                ]

            if len(transformed_names) != len(importance):
                transformed_names = [
                    f"Feature {i + 1}"
                    for i in range(len(importance))
                ]

            explanation_df = pd.DataFrame({
                "Feature": transformed_names,
                "Importance": importance,
            }).sort_values("Importance", ascending=False)

            explanation_df["Feature"] = (
                explanation_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
                .str.replace("_", " ", regex=False)
            )

            fig = px.bar(
                explanation_df.head(10).sort_values(
                    "Importance", ascending=True
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Most influential features",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Method: {explanation_method}. These values describe "
                "overall model importance, not the exact contribution "
                "of each feature to this individual prediction."
            )
        else:
            st.info(
                "This estimator does not expose built-in feature importance "
                "or coefficients."
            )


        st.subheader("🔍 SHAP Explanation")

        try:
            import shap

            # Explain the model's prediction in the transformed feature space.
            background = getattr(
                preprocessor, "feature_names_in_", None
            )

            if hasattr(estimator, "feature_importances_"):
                # TreeExplainer is suitable for tree-based estimators.
                transformed_sample = preprocessor.transform(sample)

                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(
                    transformed_sample
                )

                st.write(
                    "This explanation shows which transformed features "
                    "influenced the model's output."
                )

                # Handle SHAP output across common versions.
                values = np.asarray(shap_values)

                if values.ndim == 3:
                    # Multiclass output: select the predicted class.
                    predicted_class = list(estimator.classes_).index(
                        model.predict(sample)[0]
                    )
                    values = values[:, :, predicted_class]

                values = values.reshape(-1)

                try:
                    transformed_names = preprocessor.get_feature_names_out()
                except (AttributeError, ValueError):
                    transformed_names = [
                        f"Feature {i + 1}" for i in range(len(values))
                    ]

                if len(transformed_names) != len(values):
                    transformed_names = [
                        f"Feature {i + 1}" for i in range(len(values))
                    ]

                shap_df = pd.DataFrame({
                    "Feature": transformed_names,
                    "SHAP value": values,
                })

                shap_df["Feature"] = (
                    shap_df["Feature"]
                    .str.replace("num__", "", regex=False)
                    .str.replace("cat__", "", regex=False)
                    .str.replace("_", " ", regex=False)
                )

                shap_df["Absolute impact"] = shap_df["SHAP value"].abs()
                shap_df = shap_df.sort_values(
                    "Absolute impact", ascending=False
                ).head(10)

                fig = px.bar(
                    shap_df.sort_values("SHAP value"),
                    x="SHAP value",
                    y="Feature",
                    orientation="h",
                    title="SHAP impact for this prediction",
                    color="SHAP value",
                    color_continuous_midpoint=0,
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Plain-English interpretation of SHAP values
                st.subheader("💡 What does this mean?")

                positive_features = (
                    shap_df[shap_df["SHAP value"] > 0]
                    .sort_values("SHAP value", ascending=False)
                    .head(3)
                )

                negative_features = (
                    shap_df[shap_df["SHAP value"] < 0]
                    .sort_values("SHAP value", ascending=True)
                    .head(3)
                )

                if not positive_features.empty:
                    st.markdown("**Features increasing the model output**")

                    for _, item in positive_features.iterrows():
                        st.write(
                            f"• **{item['Feature']}**: positive SHAP "
                            f"impact of {item['SHAP value']:.4f}"
                        )
                else:
                    st.write(
                        "No positive feature contributions were found "
                        "among the displayed features."
                    )

                if not negative_features.empty:
                    st.markdown("**Features decreasing the model output**")

                    for _, item in negative_features.iterrows():
                        st.write(
                            f"• **{item['Feature']}**: negative SHAP "
                            f"impact of {item['SHAP value']:.4f}"
                        )
                else:
                    st.write(
                        "No negative feature contributions were found "
                        "among the displayed features."
                    )

                st.info(
                    "These contributions explain movement in the selected "
                    "model output relative to its baseline. They do not "
                    "prove causation, and their scale depends on the "
                    "explainer's output."
                )

                st.caption(
                    "Positive and negative SHAP values indicate movement "
                    "relative to the explainer's baseline for the selected "
                    "model output. They are not probabilities."
                )

            else:
                st.info(
                    "This first SHAP integration supports tree-based models. "
                    "Support for other estimator types can be added next."
                )

        except ImportError:
            st.warning(
                "SHAP is not installed in this Python environment. "
                "Install it with: python -m pip install shap"
            )
        except Exception as exc:
            st.warning(f"SHAP explanation could not be generated: {exc}")

        st.subheader("Input Summary")
        st.dataframe(sample, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not generate prediction: {exc}")
