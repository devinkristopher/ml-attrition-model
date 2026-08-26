import pandas as pd
import numpy as np
import pytest

from src.preprocess import impute_missing_values, encode_categoricals, introduce_missing_values, validate_dataframe, check_data_quality

def test_impute_missing_values_numerical():
    """Median fill should replace NaN values with the column median."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, np.nan, 40.0, 50.0]
    })

    # use blank Series for categorical because we are only testing numerical imputation here
    numeric_fill_values = pd.Series([35.0], index=["age"])
    result = impute_missing_values(df, numeric_fill_values, pd.Series())
    
    assert result["age"].isna().sum() == 0, "Error: There should be no missing values after filling"
    assert result["age"].iloc[2] == 35.0, "Error: Missing value should be filled with median (35.0)"

def test_impute_missing_values_categorical():
    """Mode fill should replace NaN values with the column mode."""
    df = pd.DataFrame({
        "marriage_status": ["Married", "Single", np.nan, "Married", "Divorced"]
    })

    categorical_fill_values = pd.Series(["Married"], index=["marriage_status"])
    # use blank Series for numeric because we are only testing categorical imputation here
    result = impute_missing_values(df, pd.Series(), categorical_fill_values)

    assert result["marriage_status"].isna().sum() == 0, "Error: There should be no missing values after filling"
    assert result["marriage_status"].iloc[2] == "Married", "Error: Missing value should be filled with mode ('Married'  )"


def test_encode_categoricals():
    """Test that encoding the categorical columns works as expected."""

    # new dataframe with categorical columns and sample data
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "blue"],
        "size": ["S", "M", "L", "M"]
    })

    categorical_columns = ["color", "size"]
    expected_columns = [
        "color_blue", "color_green", "color_red",
        "size_L", "size_M", "size_S"
    ]

    # dataframe, categorical_columns, expected_columns
    result = encode_categoricals(df, categorical_columns, expected_columns)

    # Check that the resulting DataFrame has the expected columns
    assert list(result.columns) == expected_columns, "Error: Encoded DataFrame does not have the expected columns"

def test_df_preservation():
    """Tests that the original dataframe is preserved throughout preprocessing, and is not modified during operations."""
    original = pd.DataFrame({
        "Age": [25, None, 40],
        "Department": ["Sales", "HR", None],
    })
    original_snapshot = original.copy(deep=True)

    # using these later on for multiple function calls; 
    # the local scope (each function call) will use the same variable name as this function-level variable assignment
    # all 5 functions of preprocess.py are called. 
    # not testing for accuracy in this function, just testing to ensure that original matches the original_snapshot above.
    numeric_columns=["Age"]
    expected_columns = [
        "Age",
        "Department_HR",
        "Department_Sales"
    ]
    categorical_columns=["Department"]
    fraction=0.5
    random_state=42
    numeric_fill_values = pd.Series({"Age": 32.5})
    categorical_fill_values = pd.Series({"Department": "HR"})

    validate_dataframe(
        dataframe=original,
        required_columns=original.columns.astype(str).tolist(),
        target_column=original.columns[0]
    )

    impute_missing_values(
    dataframe=original,
    numeric_fill_values=numeric_fill_values,
    categorical_fill_values=categorical_fill_values,
    )

    introduce_missing_values(
        dataframe=original, 
        columns=original.columns.astype(str).tolist(), 
        fraction=fraction,
        random_state=random_state
    )

    check_data_quality(
        dataframe=original,
        numeric_columns=numeric_columns
    )

    encode_categoricals(
        dataframe=original,
        categorical_columns=categorical_columns,
        expected_columns=expected_columns,
    )

    pd.testing.assert_frame_equal(original, original_snapshot)



def test_input_validation():
    """Tests that the fractional input for `introduce_missing_values()` is valid."""
    dataframe = pd.DataFrame({
        "Age": [25, None, 40],
        "Department": ["Sales", "HR", None],
    })

    invalid_fraction=103.0

    with pytest.raises(
        ValueError,
        match="Missing-value fraction",
    ):
        introduce_missing_values(
            dataframe=dataframe,
            columns=["Age"],
            fraction=invalid_fraction,
            random_state=42,
        )


def test_required_cols_error():
    """Tests that `validate_dataframe()` raises an error when a required column is missing."""
    dataframe = pd.DataFrame({
        "Age": [25, 30, 40],
        "Attrition": ["No", "Yes", "No"],
    })

    with pytest.raises(ValueError):
        validate_dataframe(
            dataframe=dataframe,
            required_columns=["Age", "Department", "Attrition"],
            target_column="Attrition",
        )
