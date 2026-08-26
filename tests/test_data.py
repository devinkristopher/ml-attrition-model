from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/employee_attrition.csv")

def load_data():
    return pd.read_csv(DATA_PATH)


def test_present_columns():
    """This test tests whether the expected columns are present in the dataframe."""
    dataframe = load_data()

    expected_columns = {
        "Age",
        "Attrition",
        "BusinessTravel",
        "DailyRate",
        "Department",
        "DistanceFromHome",
        "Education",
        "EducationField",
        "EmployeeCount",
        "EmployeeNumber",
        "EnvironmentSatisfaction",
        "Gender",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobRole",
        "JobSatisfaction",
        "MaritalStatus",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "Over18",
        "OverTime",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StandardHours",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    }

    assert expected_columns.issubset(set(dataframe.columns)), f"Error: one of the expected columns is missing in the Employee Attrition Dataset. Check that the following columns are present: \n {expected_columns}"


def test_target_compatibility():
    """Tests that the target column contains the expected values (Yes and No)."""
    dataframe = load_data()
    target_name = "Attrition"

    actual_unique_values = set(
        dataframe[target_name]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    expected_unique_values = {"yes", "no"}

    assert actual_unique_values == expected_unique_values, (f"Error: The values in dataframe['{target_name}'] do not match the values expected: {expected_unique_values}. Please ensure all rows have a Yes or No target value.")


def test_numeric_input_validation():
    """Input validation to verify that numeric features have valid values."""
    dataframe = load_data()

    # For employee attrition;
    
    expected_ranges = {
        # Legal mins and maxes should be evaluated to deter age-based discriminatory ingestion.
        # Rough sensible boundaries have been placed to allow for data to evolve, but should be adjusted to known boundaries if the application permits.
        # Legal boundaries should be evaluated to ensure the model does not discriminate against a protected class, such as those 40+ years of age.
        "Age": (18, 70),
        "DistanceFromHome": (0, 100),
        "MonthlyIncome": (0, 100_000),
        "PercentSalaryHike": (0, 100),
        "JobSatisfaction": (1, 4),
        "WorkLifeBalance": (1, 4),
        "TotalWorkingYears": (0, 60),
    }

    for column, (minimum, maximum) in expected_ranges.items():
        assert dataframe[column].between(minimum, maximum).all(), (
            f"{column} contains values outside "
            f"the expected range {minimum}–{maximum}"
        )