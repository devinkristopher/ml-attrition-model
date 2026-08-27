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

`DistanceFromHome` demonstrated the greatest drift. Its distribution shifted toward greater commuting distances, indicating that employees generally lived farther from the office in the simulated current dataset. Although all values remained within the 100-mile input-validation threshold, the change represents a meaningful workforce-level trend because proximity to the office may influence retention. The model’s reference data already included employees within these distance ranges, so this is not an entirely unfamiliar population; rather, employees living farther away now constitute a larger share of it. A substantial or rapidly emerging shift of this kind could reflect external factors such as housing costs, broader economic conditions, or migration patterns. It should therefore prompt HR awareness and further investigation, though it does not by itself indicate an immediate increase in attrition. Instead, it may change how existing attrition-risk thresholds should be interpreted. For example, if only a minority of employees now live within 15 miles of the office, whereas a majority previously did, expanded hybrid or remote-work policies could help mitigate commute-related attrition risk.

`HourlyRate`, `DailyRate`, and `MonthlyIncome` shifted upward, signifying an increase in economic presence. This could reflect compensation packages, bonuses, raises, inflationary wage increases, more hiring in higher-paying roles, or promotional trends. Without knowing the mathematical relationship between these values and how they are calculated, it would be taking liberties to derive and isolate variables from the wage-hour relationship. Therefore, the observation that employees are generally falling in higher compensation brackets is the primary takeaway.

`Age` was especially interesting (see image below) -- a once largely normal distribution with a slight right tail evolved into a markedly distinct bimodal distribution, with one generally younger bell-shaped modality and another distinct, more mature modality with more volume and presence per age bracket. While this feature did not have the most drift, it had a considerable migration of volume into unique, specific demographics that had previously little data. Some environments may have a higher likelihood of attrition when evaluating those of higher age ranges. Nevertheless, the higher age brackets certainly deserve close attention, especially considering the lack of associated training data in the original set.

<img width="654" height="146" alt="image" src="https://github.com/user-attachments/assets/d332f058-16ec-4f44-ae06-68a3a6326e24" />

`Overtime` demonstrated considerable changes. Where the majority of users originally had no overtime, now it is the case that the majority of users do have overtime. This is related to compensation, but a pronounced increase in overtime -- where the majority of users now associate with the `Overtime` subset -- deserves attention. Overtime with respect to younger users may show a different attrition rate than overtime with respect to older users. Overtime in management roles may show a materially different burden than overtime in physical, or engineering roles. In terms of HR, a more mild increase in compensation may not justify the burnout, injury risk, or emotional contagion that may be associated with a majority-overtime work culture.

`BusinessTravel` showed an interesting shift as well. Fewer users report that they don't travel at all; therefore, more users are traveling at least some for work. With the increase in sample size, there has been a proportional increase in both users traveling frequently and users traveling rarely. Also, where users were previously more likely to travel rarely, they are now more likely to travel frequently. Factored in with increased overtime and a more mild increase in compensation, changes in business travel could lead to attrition as users spend less time at home. Perhaps most importantly, we don't have much training data for frequently-flying employees for this company. While general trends certainly matter, model training principles are foundational to this evaluation; therefore, when a minority group experiences a sudden increase in volume, it is appropriate to address the class imbalance in some manner. Perhaps upsampling the minority-majority groups would be the best next step. 

In conclusion, we've seen demographic changes that are linked to attrition-sensitive regions -- compensation, workload, age, proximity to home, and time traveling away from home. These shifting demographics certainly could be of value to HR and should be presented as such. While we don't have evidence that there is a present attrition spike at this time, we do recognize model-related principle risk as some previously rare observations are now exceedingly frequent -- and in some cases, the majority case. In response to these changes, I would recommend upsampling some decreasingly rare occurrences for training purposes. Alternatively, the model could be monitored closely for performance in these key areas.

<img width="2352" height="1266" alt="image" src="https://github.com/user-attachments/assets/9d6ddaf0-db4e-4da9-94ce-ab1c22be086a" />
