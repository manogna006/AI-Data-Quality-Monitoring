import pandas as pd


def profile_dataset(df):
    """
    Generate a basic automated data profile.
    """

    profile = pd.DataFrame({
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "missing_count": df.isnull().sum().values,
        "missing_percentage": (
            df.isnull().mean() * 100
        ).values,
        "unique_values": df.nunique(dropna=True).values
    })

    return profile


def print_dataset_summary(df):

    print("\n========== DATASET SUMMARY ==========")

    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Duplicate rows    : {df.duplicated().sum()}")
    print(
        f"Total missing values: {df.isnull().sum().sum()}"
    )

    print("\nNumerical columns:")
    print(df.select_dtypes(include="number").columns.tolist())

    print("\nCategorical columns:")
    print(
        df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
    )

    print("\n======================================")


if __name__ == "__main__":

    from data_loader import load_dataset

    dataset_path = "datasets/raw/online_retail_II.xlsx"

    df = load_dataset(dataset_path)

    print_dataset_summary(df)

    profile = profile_dataset(df)

    print("\n========== COLUMN PROFILE ==========")
    print(profile.to_string(index=False))

    # Save profiling results
    output_path = "results/tables/online_retail_profile.csv"
    profile.to_csv(output_path, index=False)

    print(f"\nProfile saved to: {output_path}")