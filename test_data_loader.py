
import pandas as pd

from src.utils.data_loader import (
    get_dataset_summary,
    validate_target,
)


data = pd.DataFrame({
    "hours_studied": [2, 4, 6, 8],
    "attendance": [60, 70, 85, 95],
    "result": ["Fail", "Fail", "Pass", "Pass"],
})

summary = get_dataset_summary(data)

print("Dataset summary:")
print(summary)

assert summary["rows"] == 4
assert summary["columns"] == 3
assert validate_target(data, "result")

print("\nAll basic data-loader tests passed!")
