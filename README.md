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
```
## Drift Report

Which features showed drift and why? 

With simulated drift, **7 out of 30 (about 23.333%) features demonstrated drift**. 

| Feature | Type | Drift score | Status |
|---|---|---:|---|
| DistanceFromHome | Numeric | 3.318097 | Drift detected |
| HourlyRate | Numeric | 2.784956 | Drift detected |
| DailyRate | Numeric | 2.018343 | Drift detected |
| Age | Numeric | 1.107598 | Drift detected |
| OverTime | Categorical | 0.928754 | Drift detected |
| MonthlyIncome | Numeric | 0.770417 | Drift detected |
| BusinessTravel | Categorical | 0.721514 | Drift detected |

The features with the most prominent drift were `DistanceFromHome`, `HourlyRate`, and `DailyRate`. 

<img width="2352" height="1266" alt="image" src="https://github.com/user-attachments/assets/9d6ddaf0-db4e-4da9-94ce-ab1c22be086a" />


`DistanceFromHome` demonstrated the greatest drift. Its distribution shifted toward greater commuting distances, indicating that employees generally lived farther from the office in the simulated current dataset. Although all values remained within the 100-mile input-validation threshold, the change represents a meaningful workforce-level trend because proximity to the office may influence retention. The reference data included moderately long commutes, but the upper portion of the current range was previously unrepresented or sparsely represented. A substantial or rapidly emerging shift of this kind could reflect external factors such as housing costs, broader economic conditions, or migration patterns. It should therefore prompt HR awareness and further investigation, though it does not by itself indicate an immediate increase in attrition. For example, if considerably fewer employees now live near the office, expanded hybrid or remote-work policies could help mitigate commute-related attrition risk.

`HourlyRate`, `DailyRate`, and `MonthlyIncome` shifted upward, signifying an increase in economic presence. This could reflect compensation packages, bonuses, raises, inflationary wage increases, more hiring in higher-paying roles, or promotional trends. Without knowing the mathematical relationship between these values and how they are calculated, it would be taking liberties to derive and isolate variables from the wage-hour relationship. Therefore, the observation that employees are generally falling in higher compensation brackets is the primary takeaway.

`Age` was especially interesting (see image below). A previously unimodal distribution developed a more visibly bimodal shape, with increased representation in older age ranges. Although the reference data included older employees, these ranges were less densely represented, and the current data extends beyond the former upper range. This does not establish that older employees are more likely to leave, but it warrants evaluating model performance across age groups to ensure that predictions remain reliable for the changing workforce.

<img width="654" height="146" alt="image" src="https://github.com/user-attachments/assets/d332f058-16ec-4f44-ae06-68a3a6326e24" />

`Overtime` demonstrated considerable change. Employees reporting overtime shifted from a minority in the reference data to a majority in the current data. This pronounced increase deserves attention because sustained overtime could indicate greater workload and burnout risk. Its relationship with attrition may also vary across age groups and job roles. From an HR perspective, the simultaneous increases in compensation and overtime raise the question of whether higher pay adequately offsets the additional demands placed on employees.

`BusinessTravel` showed an interesting shift as well. Frequent travel increased from approximately 19% of the reference data to 59% of the current data, while rare travel declined from approximately 71% to 36%. Combined with increased overtime, this may indicate greater demands on employees’ time away from home. Frequent travelers were less prominent in the reference data but now constitute the majority, creating a potential model-risk concern. Model performance should therefore be evaluated specifically for this group using recent labeled outcomes.

In conclusion, we observed changes across several attrition-sensitive dimensions: compensation, workload, age, proximity to the office, and time spent traveling away from home. These changes do not prove that attrition has increased, but together they may describe a changing employee experience in which higher compensation coincides with greater demands on employees’ time and flexibility. This information could be valuable to HR and should also prompt closer model monitoring. The next step should be to evaluate performance within the shifted subgroups using recent labeled data. If performance has deteriorated, retraining on data that better represents the current workforce should be considered.
