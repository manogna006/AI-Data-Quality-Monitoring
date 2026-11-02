import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """
    Load a CSV or Excel dataset based on its file extension.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}"
        )

    return df


if __name__ == "__main__":

    dataset_path = Path(
        "datasets/raw/online_retail_II.xlsx"
    )

    df = load_dataset(dataset_path)

    print("\nDataset loaded successfully!")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())