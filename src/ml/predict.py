
import pandas as pd

from src.utils.storage import load_model


def predict_sample(
    features,
    model_name="classification_random_forest",
):
    """
    Generate predictions using a saved ModelForge pipeline.

    features can be a dictionary of column names and values,
    or a DataFrame containing one or more rows.
    """

    model = load_model(model_name)

    if isinstance(features, dict):
        input_data = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        input_data = features.copy()
    else:
        raise TypeError(
            "Features must be a dictionary or pandas DataFrame."
        )

    if input_data.empty:
        raise ValueError("Please provide at least one row to predict.")

    if input_data.isna().all(axis=1).any():
        raise ValueError("An input row cannot have every feature missing.")

    # Match the input columns and order used during training.
    preprocessor = model.named_steps["preprocessor"]
    expected_columns = list(preprocessor.feature_names_in_)

    missing_columns = [
        col for col in expected_columns
        if col not in input_data.columns
    ]

    extra_columns = [
        col for col in input_data.columns
        if col not in expected_columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            + ", ".join(missing_columns)
        )

    if extra_columns:
        raise ValueError(
            "Unexpected feature columns: "
            + ", ".join(extra_columns)
        )

    input_data = input_data[expected_columns]

    predictions = model.predict(input_data)

    estimator = model.named_steps["model"]

    if hasattr(estimator, "predict_proba"):
        probabilities_array = model.predict_proba(input_data)

        probabilities = [
            {
                str(class_name): float(probability)
                for class_name, probability in zip(
                    estimator.classes_,
                    row_probabilities,
                )
            }
            for row_probabilities in probabilities_array
        ]
    else:
        probabilities = None

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities,
        "model_name": model_name,
        "rows_predicted": len(input_data),
    }


if __name__ == "__main__":
    # Example matching the Iris dataset's four feature columns.
    sample = {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2,
    }

    result = predict_sample(sample)

    print("\nPREDICTION RESULT")
    print("-" * 30)
    print("Prediction:", result["predictions"][0])

    if result["probabilities"]:
        print("Probabilities:", result["probabilities"][0])

