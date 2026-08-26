# Employee Attrition MLOps Pipeline

An end-to-end MLOps pipeline for predicting employee attrition using the
IBM HR Analytics Employee Attrition & Performance dataset.

## Project Overview

This project demonstrates data versioning with DVC, experiment tracking
with MLflow, automated testing with pytest, CI/CD with GitHub Actions,
and data-drift monitoring with Evidently.

## Dataset

- Source: IBM HR Analytics Employee Attrition & Performance
- Task: Binary classification
- Target: `Attrition`
- Size: 1,470 rows and 35 columns
- Missing values are introduced reproducibly for preprocessing validation.

## Setup
Python 3.13.9 on macOS Tahoe 26.6, 
VS Code: 
Version: 1.128.1 (Universal)
Commit: 5264f2156cbcd7aea5fd004d29eaa10209155d66
Electron: 42.5.0
ElectronBuildId: 14525058
Chromium: 148.0.7778.271
Node.js: 24.17.0
V8: 14.8.178.33-electron.0
OS: Darwin arm64 25.6.0
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt