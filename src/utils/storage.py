
from pathlib import Path
import joblib


# Resolve paths from the ModelForge project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, model_name="random_forest"):
    """Save a trained model to the local artifacts/models folder."""

    safe_name = model_name.lower().replace(" ", "_")
    model_path = MODEL_DIR / f"{safe_name}.joblib"

    joblib.dump(model, model_path)

    return model_path


def load_model(model_name="random_forest"):
    """Load a previously saved model."""

    safe_name = model_name.lower().replace(" ", "_")
    model_path = MODEL_DIR / f"{safe_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Train and save the model first."
        )

    return joblib.load(model_path)


def list_saved_models():
    """Return the names of all saved model files."""

    if not MODEL_DIR.exists():
        return []

    return [
        path.stem
        for path in MODEL_DIR.glob("*.joblib")
        if path.is_file()
    ]


if __name__ == "__main__":
    print("Model storage directory:")
    print(MODEL_DIR)

    print("\nCurrently saved models:")
    models = list_saved_models()

    if models:
        for model_name in models:
            print("-", model_name)
    else:
        print("No models saved yet.")
