# Model Card: Clinical Risk Lab

## Model purpose

Clinical Risk Lab is a portfolio demonstration of an end-to-end tabular healthcare ML workflow. It shows preprocessing, probability calibration, threshold selection, subgroup evaluation, explainability, model persistence, and inference.

## Intended use

- Educational and portfolio demonstration
- Reproducible ML engineering examples
- Testing evaluation and calibration workflows on synthetic data

## Not intended for

- Diagnosis
- Treatment recommendations
- Patient triage
- Clinical decision support
- Any real-world medical use

## Data

The included cohort is synthetic. No patient records, protected health information, or proprietary datasets are included.

## Evaluation

The workflow separates training, validation, and held-out test data. Threshold selection is performed on validation data, while final performance is reported on the test split.

Metrics include AUROC, AUPRC, Brier score, sensitivity, specificity, and subgroup-level evaluation.

## Limitations

Synthetic data cannot reproduce the full distribution shift, missingness, measurement error, confounding, selection bias, demographic heterogeneity, or clinical workflow constraints present in real healthcare datasets.

Strong performance in this repository should therefore be interpreted only as a software and ML methodology demonstration.

## Responsible use

Any model intended for real clinical use would require appropriate data governance, external validation, prospective evaluation, fairness analysis, clinical oversight, regulatory review where applicable, and continuous monitoring after deployment.
