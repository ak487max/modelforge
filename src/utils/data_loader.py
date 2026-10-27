
from io import BytesIO

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv"}


def load_csv(uploaded_file):
    """Load and validate a CSV file uploaded through Streamlit."""

    if uploaded_file is None:
        raise ValueError("Please upload a CSV file.")

    filename = getattr(uploaded_file, "name", "dataset.csv")

    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported.")

    try:
        # Read the uploaded file without changing its contents.
        uploaded_file.seek(0)
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    if data.empty:
        raise ValueError("The uploaded CSV file is empty.")

    if len(data.columns) < 2:
        raise ValueError(
            "The dataset needs at least two columns: "
            "one or more features and one target."
        )

    if data.columns.duplicated().any():
        raise ValueError("The dataset contains duplicate column names.")

    # Remove rows that contain no data at all.
    data = data.dropna(how="all").reset_index(drop=True)

    if data.empty:
        raise ValueError("The dataset contains no usable rows.")

    return data


def get_dataset_summary(data):
    """Return useful dataset information for the dashboard."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_names": data.columns.tolist(),
        "missing_values": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
        "memory_usage_kb": round(
            data.memory_usage(deep=True).sum() / 1024, 2
        ),
        "numeric_columns": data.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist(),
    }


def validate_target(data, target_column):
    """Check whether a target column can be used for ML."""

    if target_column not in data.columns:
        raise ValueError("The selected target column does not exist.")

    if data[target_column].isna().any():
        raise ValueError(
            "The target column contains missing values. "
            "Clean these rows before training."
        )

    if data[target_column].nunique() < 2:
        raise ValueError(
            "The target column must contain at least two distinct values."
        )

    return True
