import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset


def load_config(path: str | Path) -> dict[str, Any]:
    """Load project configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_and_prepare(config: dict[str, Any]) -> pd.DataFrame:
    """Load the employee dataset and prepare features for drift analysis."""
    data_path = Path(config["data"]["raw_path"])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run 'dvc pull' before monitoring drift."
        )

    dataframe = pd.read_csv(data_path)

    # Drift monitoring focuses on model inputs, not the target.
    dataframe = dataframe.drop(
        columns=[config["data"]["target_column"]],
    )

    # Remove identifiers and constant columns excluded during training.
    dataframe = dataframe.drop(
        columns=config["data"]["drop_columns"],
        errors="ignore",
    )

    # Fill missing numeric values using medians.
    numeric_columns = dataframe.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        dataframe[column] = dataframe[column].fillna(
            dataframe[column].median()
        )

    # Fill missing categorical values using modes.
    categorical_columns = dataframe.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        dataframe[column] = dataframe[column].fillna(
            dataframe[column].mode()[0]
        )

    return dataframe


def create_reference_and_production(
    dataframe: pd.DataFrame,
    reference_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into reference and production data.

    Reference data represents the model's training environment.
    Production data represents observations arriving after deployment.
    """
    dataframe = dataframe.sample(
        frac=1,
        random_state=random_state,
    ).reset_index(drop=True)

    split_index = int(
        len(dataframe) * reference_fraction
    )

    reference_data = dataframe.iloc[
        :split_index
    ].copy()

    production_data = dataframe.iloc[
        split_index:
    ].copy()

    return reference_data, production_data


def introduce_drift(
    production_data: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """Introduce reproducible drift into selected production features."""
    production_data = production_data.copy(deep=True)
    generator = np.random.default_rng(random_state)

    # Compensation shifts upward.
    production_data["MonthlyIncome"] = (
        production_data["MonthlyIncome"]
        * generator.uniform(
            1.4,
            1.8,
            len(production_data),
        )
    )

    # Employees live farther from work.
    production_data["DistanceFromHome"] = (
        production_data["DistanceFromHome"]
        + generator.normal(
            10,
            3,
            len(production_data),
        )
    ).clip(0, 100)

    # Daily and hourly rates shift.
    production_data["DailyRate"] = (
        production_data["DailyRate"]
        * generator.uniform(
            1.3,
            1.6,
            len(production_data),
        )
    )

    production_data["HourlyRate"] = (
        production_data["HourlyRate"]
        + generator.normal(
            20,
            5,
            len(production_data),
        )
    ).clip(0, 200)

    # Simulate an influx of older employees.
    age_count = int(
        len(production_data) * 0.35
    )

    age_indices = generator.choice(
        production_data.index.to_numpy(),
        size=age_count,
        replace=False,
    )

    production_data.loc[
        age_indices,
        "Age",
    ] = generator.integers(
        50,
        66,
        size=age_count,
    )

    # Overtime becomes more common.
    overtime_count = int(
        len(production_data) * 0.60
    )

    overtime_indices = generator.choice(
        production_data.index.to_numpy(),
        size=overtime_count,
        replace=False,
    )

    production_data.loc[
        overtime_indices,
        "OverTime",
    ] = "Yes"

    # Frequent business travel becomes more common.
    travel_count = int(
        len(production_data) * 0.50
    )

    travel_indices = generator.choice(
        production_data.index.to_numpy(),
        size=travel_count,
        replace=False,
    )

    production_data.loc[
        travel_indices,
        "BusinessTravel",
    ] = "Travel_Frequently"

    return production_data


def get_drift_summary(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run Evidently drift detection and return a summary."""
    feature_threshold = config["monitoring"][
        "feature_drift_threshold"
    ]
    drift_share_threshold = config["monitoring"][
        "drift_share_threshold"
    ]

    # PSI gives every feature the same score direction:
    # drift is detected when PSI is at or above the threshold.
    report = Report(
        metrics=[
            DataDriftPreset(
                method="psi",
                threshold=feature_threshold,
                drift_share=drift_share_threshold,
            )
        ]
    )

    snapshot = report.run(
        reference_data=reference_data,
        current_data=production_data,
    )

    report_path = Path(
        config["monitoring"]["report_path"]
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # type-cast string or it won't work!
    snapshot.save_html(str(report_path))

    if not report_path.exists():
        raise FileNotFoundError(
            f"Report was not created at {report_path}"
        )

    result = snapshot.dict()

    # The first metric contains overall drift count and share.
    drift_count_metric = result["metrics"][0]
    drifted_count = int(
        drift_count_metric["value"]["count"]
    )
    drift_share = float(
        drift_count_metric["value"]["share"]
    )

    # Remaining metrics contain per-feature drift results.
    feature_metrics = result["metrics"][1:]
    drifted_features = []
    feature_details = {}

    for metric in feature_metrics:
        column = metric["config"]["column"]
        threshold = float(
            metric["config"]["threshold"]
        )
        drift_score = float(metric["value"])
        drifted = drift_score >= threshold

        feature_details[column] = {
            "drifted": drifted,
            "drift_score": round(
                drift_score,
                4,
            ),
            "threshold": threshold,
            "method": metric["config"]["method"],
        }

        if drifted:
            drifted_features.append(column)

    summary = {
        "total_features": len(feature_metrics),
        "drifted_features": drifted_count,
        "drift_share": round(drift_share, 3),
        "drift_share_threshold": drift_share_threshold,
        "dataset_drift": (
            drift_share >= drift_share_threshold
        ),
        "drifted_feature_names": drifted_features,
        "features": feature_details,
    }

    results_path = Path(
        config["monitoring"]["results_path"]
    )
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        f"HTML report saved to {report_path}"
    )
    print(
        f"JSON summary saved to {results_path}"
    )

    return summary


def main() -> None:
    """Run the complete drift-monitoring workflow."""
    config = load_config(
        "configs/config.yaml"
    )
    random_state = config["project"][
        "random_seed"
    ]

    print("Loading employee attrition data...")
    dataframe = load_and_prepare(config)
    print(
        f"Total rows: {len(dataframe)}"
    )

    print(
        "\nSplitting reference and production data..."
    )
    reference_data, production_data = (
        create_reference_and_production(
            dataframe=dataframe,
            reference_fraction=config[
                "monitoring"
            ]["reference_fraction"],
            random_state=random_state,
        )
    )

    print(
        "Introducing simulated production drift..."
    )
    production_data = introduce_drift(
        production_data=production_data,
        random_state=random_state,
    )

    print(
        f"\nReference data:  {len(reference_data)} rows"
    )
    print(
        f"Production data: {len(production_data)} rows"
    )

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION DATA DRIFT REPORT")
    print("=" * 60)

    summary = get_drift_summary(
        reference_data=reference_data,
        production_data=production_data,
        config=config,
    )

    print(
        f"\nFeatures drifted: "
        f"{summary['drifted_features']}/"
        f"{summary['total_features']} "
        f"({summary['drift_share'] * 100:.1f}%)"
    )

    if summary["drifted_feature_names"]:
        print("\nDrifted features:")

        for feature in summary[
            "drifted_feature_names"
        ]:
            details = summary["features"][
                feature
            ]
            print(
                f"  {feature}: "
                f"score = "
                f"{details['drift_score']} "
                f"(threshold = "
                f"{details['threshold']})"
            )
    else:
        print(
            "\nNo features showed significant drift."
        )

    if summary["dataset_drift"]:
        print(
            f"\nCRITICAL: "
            f"{summary['drift_share'] * 100:.1f}% "
            "of features drifted."
        )
        print(
            "Action required: investigate and "
            "consider retraining."
        )
        sys.exit(1)

    print(
        "\nAll clear. Feature distributions "
        "are stable."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()