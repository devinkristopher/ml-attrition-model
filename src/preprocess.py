"""Data validation and preprocessing utilities."""

from typing import Any

import numpy as np
import pandas as pd


def validate_dataframe(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    target_column: str,
) -> bool:
    """Check that a dataframe meets basic requirements."""
    if dataframe.empty:
        raise ValueError("Dataframe is empty")

    missing = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    return True


def introduce_missing_values(
    dataframe: pd.DataFrame,
    columns: list[str],
    fraction: float,
    random_state: int,
) -> pd.DataFrame:
    """Return a copy with missing values introduced reproducibly."""
    if not 0 <= fraction < 1:
        raise ValueError("Missing-value fraction must be between 0 and 1.")

    result = dataframe.copy(deep=True)
    missing_columns = [column for column in columns if column not in result.columns]

    if missing_columns:
        raise ValueError(f"Missing configured columns: {missing_columns}")

    generator = np.random.default_rng(random_state)
    number_missing = int(len(result) * fraction)

    for column in columns:
        selected_indices = generator.choice(
            result.index.to_numpy(),
            size=number_missing,
            replace=False,
        )
        result.loc[selected_indices, column] = np.nan

    return result


def check_data_quality(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> dict[str, Any]:
    """Return a dictionary of data-quality metrics."""
    total_cells = dataframe.shape[0] * dataframe.shape[1]
    total_nulls = int(dataframe.isnull().sum().sum())
    null_percentage = 0.0 if total_cells == 0 else total_nulls / total_cells * 100

    report: dict[str, Any] = {
        "total_rows": len(dataframe),
        "total_nulls": total_nulls,
        "null_percentage": round(null_percentage, 2),
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }

    for column in numeric_columns:
        if column in dataframe.columns:
            report[f"{column}_min"] = float(dataframe[column].min())
            report[f"{column}_max"] = float(dataframe[column].max())

    return report


def impute_missing_values(
    dataframe: pd.DataFrame,
    numeric_fill_values: pd.Series,
    categorical_fill_values: pd.Series,
) -> pd.DataFrame:
    """Fill missing values using statistics learned from training data."""
    result = dataframe.copy(deep=True)

    result[numeric_fill_values.index] = result[
        numeric_fill_values.index
    ].fillna(numeric_fill_values)

    for column, fill_value in categorical_fill_values.items():
        result[column] = result[column].fillna(fill_value)

    return result


def encode_categoricals(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot encode categoricals and optionally match a learned schema."""
    encoded = pd.get_dummies(
        dataframe,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    if expected_columns is not None:
        encoded = encoded.reindex(
            columns=expected_columns,
            fill_value=0,
        )

    return encoded