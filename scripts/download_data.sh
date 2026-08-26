#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

curl -L \
  "https://www.kaggle.com/api/v1/datasets/download/pavansubhasht/ibm-hr-analytics-attrition-dataset" \
  -o data/raw/employee_attrition.zip

unzip -o data/raw/employee_attrition.zip -d data/raw
rm data/raw/employee_attrition.zip

mv data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv \
   data/raw/employee_attrition.csv

ls -lh data/raw