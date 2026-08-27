# Employee Attrition MLOps Pipeline

An end-to-end MLOps pipeline for predicting employee attrition using the IBM HR Analytics Employee Attrition & Performance dataset.

## Project Overview

This project demonstrates the infrastructure surrounding a machine-learning model: data versioning with DVC, experiment tracking with MLflow, automated testing with pytest, model-performance gates, and data-drift monitoring with Evidently.

The classifier itself is intentionally straightforward. The project emphasizes reproducibility, validation, experiment comparison, artifact creation, and production monitoring.

## Dataset

- **Source:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **Task:** Binary classification
- **Target:** `Attrition` (`Yes` or `No`)
- **Size:** 1,470 rows and 35 columns
- **DVC pointer:** `data/raw/employee_attrition.csv.dvc`

The source dataset is complete, so the training pipeline reproducibly introduces missing values into `MonthlyIncome`, `DistanceFromHome`, `JobRole`, and `BusinessTravel`. Imputation values are learned from the training split only, which prevents information from the test set from leaking into preprocessing.

## Repository Structure

```text
ml-attrition-model/
├── configs/
│   └── config.yaml
├── data/
│   └── raw/
│       └── employee_attrition.csv.dvc
├── reports/
│   ├── drift_check_result.json
│   └── drift_report.html
├── scripts/
│   └── download_data.sh
├── src/
│   ├── __init__.py
│   ├── compare_experiments.py
│   ├── evaluate.py
│   ├── monitor_drift.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_preprocess.py
├── .dvcignore
├── .gitignore
├── README.md
└── requirements.txt
```

Generated model, metric, MLflow, and HTML-report artifacts are excluded from Git by `.gitignore`.

## Setup

The project was developed with Python 3.13.9 on macOS Tahoe 26.6 (`arm64`) using Visual Studio Code 1.128.1. The exact operating system and editor are not required.

Clone the repository:

```bash
git clone https://github.com/devinkristopher/ml-attrition-model.git
cd ml-attrition-model
```

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Retrieve the Dataset

The training dataset is tracked by DVC rather than Git:

```bash
dvc pull
```

The configured DVC remote is a local filesystem remote named `storage`. On the original development machine it resolves to `../ml-attrition-dvc-storage` relative to the repository's parent directory.

If that local DVC storage directory is unavailable, the included retrieval script downloads the public Kaggle dataset without requiring Kaggle CLI authentication:

```bash
bash scripts/download_data.sh
```

Both methods create:

```text
data/raw/employee_attrition.csv
```

## Configuration

`configs/config.yaml` defines the dataset path, reproducibility seed, missing-value simulation, model settings, evaluation thresholds, artifact paths, monitoring thresholds, and MLflow experiment settings.

The active model is a balanced logistic-regression classifier using the `liblinear` solver. The active YAML value for `C` is `1.0`.

## Testing

Run the complete test suite from the repository root:

```bash
pytest tests/ -v
```

The suite contains 11 tests across three levels:

- Six preprocessing unit tests covering numeric and categorical imputation, one-hot encoding, preservation of the original dataframe, invalid fractions, and missing required columns.
- Three data-validation tests covering the expected schema, binary target values, and plausible numeric ranges.
- Two model-validation tests covering prediction type and shape and a minimum F1 score on a reproducible synthetic classification dataset.

## Training and Evaluation

Run the complete training workflow from the repository root:

```bash
python -m src.train
```

To use another configuration file:

```bash
python -m src.train --config path/to/config.yaml
```

The workflow:

1. Loads and validates the employee dataset.
2. Removes identifier and constant columns configured under `data.drop_columns`.
3. Reproducibly introduces missing values into selected features.
4. Creates a stratified 80/20 training and test split.
5. Learns imputation values from the training split.
6. One-hot encodes categorical features and aligns the test schema.
7. Trains a balanced logistic-regression classifier.
8. Calculates accuracy, precision, recall, F1, and ROC-AUC.
9. Logs parameters, metrics, the data-version label, and the model to MLflow.
10. Saves the complete inference bundle to `models/employee_attrition.joblib` and metrics to `metrics/results.json`.
11. Exits with status `1` if any configured minimum threshold is missed.

The configured gates are:

| Metric | Minimum |
|---|---:|
| ROC-AUC | 0.65 |
| F1 | 0.35 |
| Recall | 0.50 |

## Current Model Results

The recorded model artifact produced:

| Metric | Score |
|---|---:|
| Accuracy | 0.7619 |
| Precision | 0.3678 |
| Recall | 0.6809 |
| F1 | 0.4776 |
| ROC-AUC | 0.7980 |

The run used 1,176 training rows, 294 test rows, and 51 encoded input features. All configured performance gates passed.

Accuracy is not sufficient by itself for this imbalanced target. Recall describes how often the classifier identifies employees who leave, while precision describes how often a positive attrition prediction is correct. F1 balances those two behaviors, and ROC-AUC evaluates ranking performance across decision thresholds.

## MLflow Experiment Tracking

Training uses the local `mlruns/` filesystem store and the `employee-attrition` experiment. Each run logs:

- Model type and logistic-regression parameters
- Random seed and test size
- Missing-value fraction
- Descriptive data-version label (`ibm-attrition-v1`)
- Accuracy, precision, recall, F1, and ROC-AUC
- The trained scikit-learn model and its inferred signature

Run training repeatedly with the desired configuration changes, then compare completed runs with:

```bash
python -m src.compare_experiments
```

The comparison script uses `mlflow.search_runs()`, orders runs by ROC-AUC, prints the top five, and identifies the best completed run.

To inspect the local tracking store in a browser:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui \
  --backend-store-uri ./mlruns \
  --port 5001
```

Then open `http://127.0.0.1:5001`.

## Drift Monitoring

Run the monitoring workflow from the repository root:

```bash
python -m src.monitor_drift
```

The script removes the target and training-excluded columns, shuffles the remaining features, uses 60% as reference data, and uses 40% as simulated production data. It then introduces reproducible changes in compensation, commuting distance, age, overtime, and business travel.

Evidently evaluates all 30 features using Population Stability Index (PSI). A feature is considered drifted at a PSI threshold of `0.20`, and dataset drift is raised when at least 20% of features drift.

Outputs are saved to:

- `reports/drift_report.html` — interactive Evidently report
- `reports/drift_check_result.json` — machine-readable summary

The script exits with status `1` when dataset drift exceeds the configured limit. Because this demonstration deliberately creates strong production drift, that nonzero status is the expected monitoring alert.


`DistanceFromHome` demonstrated the greatest drift. Its distribution shifted toward greater commuting distances, indicating that employees generally lived farther from the office in the simulated current dataset. Although all values remained within the 100-mile input-validation threshold, the change represents a meaningful workforce-level trend because proximity to the office may influence retention. The reference data included moderately long commutes, but the upper portion of the current range was previously unrepresented or sparsely represented. A substantial or rapidly emerging shift of this kind could reflect external factors such as housing costs, broader economic conditions, or migration patterns. It should therefore prompt HR awareness and further investigation, though it does not by itself indicate an immediate increase in attrition. For example, if considerably fewer employees now live near the office, expanded hybrid or remote-work policies could help mitigate commute-related attrition risk.

`HourlyRate`, `DailyRate`, and `MonthlyIncome` shifted upward, signifying an increase in economic presence. This could reflect compensation packages, bonuses, raises, inflationary wage increases, more hiring in higher-paying roles, or promotional trends. Without knowing the mathematical relationship between these values and how they are calculated, it would be taking liberties to derive and isolate variables from the wage-hour relationship. Therefore, the observation that employees are generally falling in higher compensation brackets is the primary takeaway.

`Age` was especially interesting (see image below). A previously unimodal distribution developed a more visibly bimodal shape, with increased representation in older age ranges. Although the reference data included older employees, these ranges were less densely represented, and the current data extends beyond the former upper range. This does not establish that older employees are more likely to leave, but it warrants evaluating model performance across age groups to ensure that predictions remain reliable for the changing workforce.

<img width="654" height="146" alt="image" src="https://github.com/user-attachments/assets/d332f058-16ec-4f44-ae06-68a3a6326e24" />

`Overtime` demonstrated considerable change. Employees reporting overtime shifted from a minority in the reference data to a majority in the current data. This pronounced increase deserves attention because sustained overtime could indicate greater workload and burnout risk. Its relationship with attrition may also vary across age groups and job roles. From an HR perspective, the simultaneous increases in compensation and overtime raise the question of whether higher pay adequately offsets the additional demands placed on employees.

`BusinessTravel` showed an interesting shift as well. Frequent travel increased from approximately 19% of the reference data to 59% of the current data, while rare travel declined from approximately 71% to 36%. Combined with increased overtime, this may indicate greater demands on employees’ time away from home. Frequent travelers were less prominent in the reference data but now constitute the majority, creating a potential model-risk concern. Model performance should therefore be evaluated specifically for this group using recent labeled outcomes.

In conclusion, we observed changes across several attrition-sensitive dimensions: compensation, workload, age, proximity to the office, and time spent traveling away from home. These changes do not prove that attrition has increased, but together they may describe a changing employee experience in which higher compensation coincides with greater demands on employees’ time and flexibility. This information could be valuable to HR and should also prompt closer model monitoring. The next step should be to evaluate performance within the shifted subgroups using recent labeled data. If performance has deteriorated, retraining on data that better represents the current workforce should be considered.
