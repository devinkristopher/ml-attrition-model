import os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow

# Connect to the same local tracking store used by train.py
mlflow.set_tracking_uri("file:./mlruns")

# Connect to the employee attrition experiment
experiment = mlflow.get_experiment_by_name(
    "employee-attrition"
)

if experiment is None:
    raise ValueError(
        "Experiment 'employee-attrition' was not found."
    )


# Search completed runs, sorted by ROC-AUC
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.roc_auc DESC"],
)

if runs.empty:
    raise ValueError(
        "No completed experiment runs were found."
    )


# Show the top five runs
print("Top 5 Runs by ROC-AUC:")
print("=" * 80)

for _, row in runs.head(5).iterrows():
    print(f"\nRun:       {row['run_id'][:8]}...")
    print(f"  Model:   {row['params.model_type']}")
    print(f"  C:       {row['params.C']}")
    print(f"  ROC-AUC: {row['metrics.roc_auc']:.4f}")
    print(f"  F1:      {row['metrics.f1']:.4f}")
    print(f"  Recall:  {row['metrics.recall']:.4f}")


# Identify the best run
best_run = runs.iloc[0]

print(f"\n{'=' * 80}")
print("BEST MODEL")
print("=" * 80)
print(f"Run ID:     {best_run['run_id']}")
print(f"Model Type: {best_run['params.model_type']}")
print(f"C:          {best_run['params.C']}")
print(f"ROC-AUC:    {best_run['metrics.roc_auc']:.4f}")
print(f"F1 Score:   {best_run['metrics.f1']:.4f}")
print(f"Recall:     {best_run['metrics.recall']:.4f}")
print(f"Accuracy:   {best_run['metrics.accuracy']:.4f}")
print(f"Precision:   {best_run['metrics.precision']:.4f}")