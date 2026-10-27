
from src.ml.train import train_model
from src.utils.experiment_tracker import save_experiment

models_to_train = [
    "Gradient Boosting",
    "Logistic Regression",
]

for model_name in models_to_train:
    print(f"\nTraining {model_name}...")

    model, metrics = train_model(
        data=None,
        target_column="target",
        model_name=model_name,
        test_size=0.2,
        task="classification",
    )

    save_experiment(
        model_name=model_name,
        dataset_name="Iris",
        task="classification",
        metrics=metrics,
        target_column="target",
    )

    print(f"{model_name} trained successfully!")

    for metric in ("accuracy", "precision", "recall", "f1_score"):
        if metric in metrics:
            print(f"{metric}: {metrics[metric]:.4f}")

print("\nAll comparison experiments are saved!")
