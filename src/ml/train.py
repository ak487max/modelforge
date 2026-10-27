
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target

from src.utils.storage import save_model


CLASSIFICATION_MODELS = {
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Logistic Regression": LogisticRegression,
}

REGRESSION_MODELS = {
    "Random Forest": RandomForestRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "Linear Regression": LinearRegression,
}



def prepare_data(data, target_column):
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    # Remove rows where the target is missing
    data = data.dropna(subset=[target_column]).copy()

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove feature columns that contain only missing values
    X = X.dropna(axis=1, how="all")

    # Ensure at least one usable feature exists
    if X.empty or X.shape[1] == 0:
        raise ValueError("The dataset has no usable feature columns.")

    # Detect the task
    from sklearn.utils.multiclass import type_of_target

    target_type = type_of_target(y)

    if target_type in ("binary", "multiclass"):
        detected_task = "classification"
    else:
        detected_task = "regression"

    return X, y, detected_task


def build_preprocessor(X):
    """Build preprocessing steps using training data column types."""

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    unsupported_columns = [
        col for col in X.columns
        if col not in numeric_columns + categorical_columns
    ]

    if unsupported_columns:
        raise ValueError(
            "Unsupported feature types in columns: "
            + ", ".join(unsupported_columns)
        )

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers=transformers)


def train_model(
    data=None,
    target_column=None,
    model_name="Random Forest",
    test_size=0.2,
    task=None,
):
    """
    Train a classification or regression model.

    When data is omitted, use the built-in Iris classification dataset.
    """

    if data is None:
        from sklearn.datasets import load_iris

        dataset = load_iris(as_frame=True)
        data = dataset.data.copy()
        data["target"] = dataset.target
        target_column = "target"
        task = "classification"

    if target_column is None:
        raise ValueError("Please specify a target column.")

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    X, y, detected_task = prepare_data(data, target_column)

    if task is None or task == "auto":
        task = detected_task

    if task not in ("classification", "regression"):
        raise ValueError(
            "Task must be classification or regression."
        )

    if task == "classification":
        model_options = CLASSIFICATION_MODELS
   

        if y.nunique() < 2:
            raise ValueError("Classification needs at least two classes.")

        # Stratification preserves class proportions when possible.
        stratify = y if y.value_counts().min() >= 2 else None
    else:
        model_options = REGRESSION_MODELS
        stratify = None

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression requires a numeric target.")

    if model_name not in model_options:
        raise ValueError(
            f"Unknown model for {task}: {model_name}. "
            f"Available: {list(model_options)}"
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to split this dataset. Check its size and class counts."
        ) from exc

    preprocessor = build_preprocessor(X_train)

    model_params = {"random_state": 42}

    if model_name == "Random Forest":
        model_params["n_estimators"] = 100

    if model_name == "Logistic Regression":
        model_params["max_iter"] = 2000

    estimator = model_options[model_name](**model_params)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    print("Selected task:", task)
    print("Target column:", target_column)
    print("Target dtype:", y_train.dtype)
    print("Target values:", y_train.unique())
    print("Estimator:", type(estimator).__name__)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions,
                    average="weighted", zero_division=0
                )
            ),
        }
    else:
        mse = mean_squared_error(y_test, predictions)

        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(mse ** 0.5),
            "r2_score": float(r2_score(y_test, predictions)),
        }

    # Save a unique model file for each task/model combination.
    artifact_name = f"{task}_{model_name.lower().replace(' ', '_')}"
    model_path = save_model(model, artifact_name)

    metrics.update({
        "model_name": model_name,
        "task": task,
        "target_column": target_column,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X.columns.tolist(),
        "classes": (
            [str(value) for value in model.named_steps["model"].classes_]
            if task == "classification"
            else []
        ),
        "model_path": str(model_path),
    })

    return model, metrics


if __name__ == "__main__":
    model, results = train_model()

    print("\nMODEL TRAINING COMPLETE")
    print("-" * 35)

    for key, value in results.items():
        print(f"{key}: {value}")
