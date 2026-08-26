"""Train and evaluate the employee attrition model."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .evaluate import calculate_metrics, thresholds_passed
from .preprocess import (
    check_data_quality,
    encode_categoricals,
    impute_missing_values,
    introduce_missing_values,
    validate_dataframe,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the employee attrition dataset."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataframe = pd.read_csv(data_path)
    print(f"Loaded {len(dataframe)} rows and {len(dataframe.columns)} columns")
    return dataframe


def save_artifacts(
    artifact: dict[str, Any],
    metrics: dict[str, float | int],
    config: dict[str, Any],
) -> None:
    """Save the model, preprocessing metadata, and evaluation metrics."""
    model_path = Path(config["artifacts"]["model_path"])
    metrics_path = Path(
        config["artifacts"].get("metrics_path", "metrics/results.json")
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Model bundle saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


def build_model(
    parameters: dict[str, Any],
    random_state: int,
) -> LogisticRegression:
    """Create the configured classification model."""
    model_parameters = parameters.copy()
    model_parameters.setdefault("random_state", random_state)

    return LogisticRegression(**model_parameters)


def train_model(
    config_path: str | Path = "configs/config.yaml",
) -> tuple[dict[str, float | int], dict[str, float]]:
    """Run the complete training workflow."""
    config = load_config(config_path)
    seed = config["project"]["random_seed"]
    target = config["data"]["target_column"]

    dataframe = load_data(config["data"]["raw_path"])
    validate_dataframe(dataframe, [target], target)

    unexpected_targets = set(dataframe[target].dropna().unique()) - {"No", "Yes"}
    if unexpected_targets or dataframe[target].isna().any():
        raise ValueError(
            "Attrition must contain only non-null 'No' and 'Yes' values; "
            f"found unexpected values: {sorted(unexpected_targets)}"
        )

    dataframe = dataframe.drop(
        columns=config["data"]["drop_columns"],
        errors="ignore",
    )
    features = dataframe.drop(columns=[target])
    labels = dataframe[target].map({"No": 0, "Yes": 1})

    if config["missing_values"]["simulate"]:
        features = introduce_missing_values(
            dataframe=features,
            columns=config["missing_values"]["columns"],
            fraction=config["missing_values"]["fraction"],
            random_state=seed,
        )

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    quality = check_data_quality(features, numeric_columns)
    print(
        f"Data quality: {quality['total_nulls']} nulls, "
        f"{quality['duplicate_rows']} duplicate rows"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config["data"]["test_size"],
        random_state=seed,
        stratify=labels,
    )

    # Learn every preprocessing value from the training split only.
    numeric_fill_values = X_train[numeric_columns].median()
    categorical_fill_values = X_train[categorical_columns].mode().iloc[0]

    X_train = impute_missing_values(
        X_train,
        numeric_fill_values,
        categorical_fill_values,
    )
    X_test = impute_missing_values(
        X_test,
        numeric_fill_values,
        categorical_fill_values,
    )

    X_train = encode_categoricals(X_train, categorical_columns)
    encoded_columns = X_train.columns.tolist()
    X_test = encode_categoricals(
        X_test,
        categorical_columns,
        expected_columns=encoded_columns,
    )

    model = build_model(
    parameters=config["model"]["parameters"],
    random_state=seed,
    )
    print("Training logistic regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, float | int] = calculate_metrics(
        y_true=y_test,
        y_pred=predictions,
        y_probability=probabilities,
    )
    metrics.update(
        {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "input_features": len(encoded_columns),
        }
    )

    print("\nResults")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")

    artifact = {
        "model": model,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "encoded_columns": encoded_columns,
        "target_column": target,
    }
    save_artifacts(artifact, metrics, config)

    return metrics, config["evaluation"]["minimum_thresholds"]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training and enforce performance thresholds."""
    arguments = parse_arguments()
    metrics, thresholds = train_model(arguments.config)

    if not thresholds_passed(metrics, thresholds):
        sys.exit(1)

    print("\nAll thresholds passed!")


if __name__ == "__main__":
    main()
