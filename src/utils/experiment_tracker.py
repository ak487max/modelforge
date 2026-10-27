
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "artifacts" / "experiments"
EXPERIMENT_FILE = EXPERIMENT_DIR / "history.json"


def _read_history():
    """Load experiment history from disk."""

    if not EXPERIMENT_FILE.exists():
        return []

    try:
        with EXPERIMENT_FILE.open("r", encoding="utf-8") as file:
            history = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Could not read experiment history: {exc}"
        ) from exc

    if not isinstance(history, list):
        raise RuntimeError("Experiment history must be a JSON list.")

    return history


def save_experiment(
    model_name,
    dataset_name,
    task,
    metrics,
    target_column=None,
):
    """Save one training run and return its experiment ID."""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Store only serializable evaluation data, not the model object.
    safe_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "model"
    }

    record = {
        "experiment_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": str(model_name),
        "dataset_name": str(dataset_name),
        "task": str(task),
        "target_column": (
            str(target_column) if target_column is not None else None
        ),
        "metrics": safe_metrics,
    }

    history = _read_history()
    history.append(record)

    temporary_file = EXPERIMENT_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, allow_nan=False)

    temporary_file.replace(EXPERIMENT_FILE)

    return record["experiment_id"]


def get_experiment_history():
    """Return all saved experiments, newest first."""

    history = _read_history()

    return sorted(
        history,
        key=lambda item: item.get("timestamp", ""),
        reverse=True,
    )


def clear_experiment_history():
    """Delete all saved experiment records."""

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    if EXPERIMENT_FILE.exists():
        EXPERIMENT_FILE.unlink()


if __name__ == "__main__":
    print("Experiment history file:")
    print(EXPERIMENT_FILE)

    history = get_experiment_history()
    print(f"\nSaved experiments: {len(history)}")

    for experiment in history:
        print(
            experiment["timestamp"],
            "|",
            experiment["model_name"],
            "|",
            experiment["task"],
        )
