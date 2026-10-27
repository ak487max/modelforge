
from src.ml.train import train_model


def evaluate_model(
    data=None,
    target_column=None,
    model_name="Random Forest",
    test_size=0.2,
    task=None,
):
    """
    Train and evaluate a model on a held-out test set.

    Supports classification and regression.
    """

    model, metrics = train_model(
        data=data,
        target_column=target_column,
        model_name=model_name,
        test_size=test_size,
        task=task,
    )

    return {
        **metrics,
        "model": model,
    }


if __name__ == "__main__":
    results = evaluate_model()

    print("\nMODEL EVALUATION RESULTS")
    print("-" * 35)

    for key, value in results.items():
        if key == "model":
            continue

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
