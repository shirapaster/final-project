import pandas as pd
from pandas import DataFrame


def load_data(file_path: str) -> DataFrame:
    """Load the dataset from the specified file path."""
    return pd.read_csv(file_path)


def inspect_data(data: DataFrame) -> DataFrame:
    """Provide an overview of the dataset."""
    print("Data Overview:")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print("Data types and missing values per column:")
    print(data.dtypes)
    print(data.isnull().sum())
    print("Statistical summary of numeric columns:")
    print(data.describe())
    return data


def drop_columns_with_many_missing(data: DataFrame, threshold: float = 0.5) -> DataFrame:
    """Drop columns with more than the specified threshold of missing values."""
    missing_values = data.isnull().sum()
    columns_to_drop = missing_values[missing_values > len(data) * threshold].index
    print("Columns with more than 50% missing values (to be dropped):")
    print(columns_to_drop)
    return data.drop(columns=columns_to_drop)


def fill_missing_values_by_group(
    data: DataFrame, columns: list[str], group_by_columns: list[str]
) -> DataFrame:
    """Fill missing values in specified columns based on group mean."""
    for column in columns:
        if data[column].isnull().sum() > 0:
            data[column] = data.groupby(group_by_columns)[column].transform(
                lambda x: x.fillna(x.mean())
            )
    return data


def detect_outliers(data: DataFrame, column: str, factor: float = 3) -> DataFrame:
    """Detect outliers in a specified column based on the factor*IQR rule."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]


def remove_outliers(data: DataFrame, columns: list[str], factor: float = 3) -> DataFrame:
    """Remove outliers from specified columns."""
    for column in columns:
        outliers = detect_outliers(data, column, factor)
        print(f"{len(outliers)} outliers removed from column {column}.")
        data = data[~data.index.isin(outliers.index)]
    return data


def check_group_balance(data: DataFrame) -> None:
    """Check if all group combinations exist and count rows per group."""
    print("\nCounts per group (Genotype and Treatment):")
    group_counts = data.groupby(['Genotype', 'Treatment']).size()
    print(group_counts)

    # Check for missing group combinations
    expected_combinations = [(g, t) for g in ['Control', 'Ts65Dn'] for t in ['Saline', 'Memantine']]
    missing_combinations = [combo for combo in expected_combinations if combo not in group_counts.index]
    if missing_combinations:
        print(f"Warning: Missing group combinations: {missing_combinations}")
    else:
        print("All expected group combinations are present.")


def save_cleaned_data(data: DataFrame, file_path: str) -> None:
    """Save the cleaned dataset to the specified path."""
    data.to_csv(file_path, index=False)
    print(f"Cleaned and relevant data saved to: {file_path}")



