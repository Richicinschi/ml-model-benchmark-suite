"""Export sklearn built-in datasets to CSV for benchmarking."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits, load_iris, load_diabetes


def save_dataset(name: str, data, target_col: str = "target") -> Path:
    path = Path(__file__).parent / f"{name}.csv"
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[target_col] = data.target
    df.to_csv(path, index=False)
    print(f"Saved {path} with shape {df.shape}")
    return path


if __name__ == "__main__":
    save_dataset("iris", load_iris(as_frame=False))
    save_dataset("digits", load_digits(as_frame=False))
    save_dataset("diabetes", load_diabetes(as_frame=False))
