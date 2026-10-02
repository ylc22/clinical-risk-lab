# Clinical Risk Lab 🩺📈

**An end-to-end machine-learning project for calibrated, explainable clinical risk prediction from tabular health data.**

[![CI](https://github.com/ylc22/clinical-risk-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/ylc22/clinical-risk-lab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Why this project

Healthcare ML should be treated as a **decision system**, not just an AUC competition. This repository demonstrates a complete modeling workflow: leakage-safe preprocessing, model training, probability calibration, threshold selection, subgroup auditing, explainability, reproducibility, testing, and command-line inference.

### Workflow

```text
patient features
   ↓
missing-value handling + encoding + scaling
   ↓
logistic risk model
   ↓
probability calibration
   ↓
validation-set threshold selection
   ↓
held-out evaluation
   ↓
subgroup audit + feature importance
   ↓
reusable inference artifact
```

## Highlights

- Synthetic cohort generator with nonlinear clinical risk structure
- Leakage-safe `Pipeline` and `ColumnTransformer`
- Logistic regression with class balancing
- Isotonic probability calibration
- AUROC, AUPRC, Brier score, sensitivity, and specificity
- Validation-only threshold selection using Youden's J
- Performance audits by sex and smoking status
- Permutation feature importance
- Saved model artifact and single-patient CLI inference
- Unit tests and GitHub Actions CI

## Quickstart

```bash
git clone https://github.com/ylc22/clinical-risk-lab.git
cd clinical-risk-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_demo.py
```

Then run a single-patient inference example:

```bash
python scripts/predict.py \
  --age 67 \
  --bmi 31 \
  --sbp 152 \
  --a1c 7.4 \
  --hdl 42 \
  --smoker yes \
  --sex F
```

## Outputs

```text
outputs/
├── metrics.json
├── calibration.png
├── feature_importance.csv
└── model.joblib
```

## Why calibration matters

A model can rank patients correctly while still producing misleading probabilities. In a risk-setting workflow, a prediction near 30% should correspond to roughly 30% observed event incidence among comparable predictions. This project therefore evaluates both **discrimination** and **calibration**.

## Evaluation design

The synthetic cohort is split into train, validation, and test partitions. Model fitting occurs on the training set, threshold selection occurs on validation data, and final metrics are reported on a held-out test set. This avoids tuning directly on the final evaluation cohort.

## Subgroup auditing

The demo reports the same core performance metrics separately across selected demographic/behavioral groups. These checks are not a substitute for a full fairness or clinical validation study, but they demonstrate the engineering pattern required to avoid relying on aggregate performance alone.

## Repository structure

```text
src/clinical_risk_lab/  reusable modeling package
scripts/train_demo.py   train, calibrate, evaluate, save
scripts/predict.py      command-line inference
tests/                  unit tests
outputs/                generated model artifacts and reports
```

## Tech stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `joblib` · `pytest` · `GitHub Actions`

## Responsible-use note

The included cohort is synthetic. This project is an ML engineering demonstration only; it is **not a medical device, diagnostic model, or clinical decision-support system** and should not be used for patient care.

## License

MIT
