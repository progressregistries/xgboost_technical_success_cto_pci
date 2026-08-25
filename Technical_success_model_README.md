# Technical Success Model

This folder contains a cleaned, generic version of the technical-success model
development pipeline.

## Files

- `Technical_success_model.py` - full pipeline implementation.
- `Technical_success_model.ipynb` - compact notebook runner for the Python file.
- `Technical_success_model_requirements.txt` - Python dependencies.

## What The Pipeline Does

1. Loads a local CSV through a user-filled generic column map.
2. Recodes binary, categorical, and continuous predictors.
3. Creates a stratified train/test split.
4. Applies the training-only missingness exclusion rule.
5. Performs construct-aware de-duplication.
6. Ranks candidate predictors by bootstrap permutation importance.
7. Uses LASSO plus optional Boruta for 5-variable selection.
8. Optionally runs model-specific one-for-one swap search.
9. Tunes and compares model families.
10. Calibrates the representative model with isotonic calibration.
11. Exports the calibrated model artifact.
12. Compares against J-CTO, PROGRESS-CTO, and CASTLE scores.
13. Runs held-out fairness metrics.
14. Optionally runs leave-one-center-out validation.

## Before Running

Open `Technical_success_model.py` and replace every `TODO_*` value in `CONFIG`
with the matching column from your local dataset.

