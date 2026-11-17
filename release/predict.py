"""Predict from complete clinical inputs. Run from the release directory."""

import json
import numpy as np
import pandas as pd
import xgboost as xgb


def predict(inputs):
    """Return success and failure probabilities; reject incomplete/invalid input."""
    with open("preprocessing.json", encoding="utf-8") as handle:
        spec = json.load(handle)
    names = [entry["name"] for entry in spec["inputs"]]
    if len(inputs.columns) != 5 or set(inputs.columns) != set(names):
        raise ValueError("CSV must contain exactly these five columns: " + ", ".join(names))
    if inputs.empty:
        raise ValueError("CSV must contain at least one complete procedure.")
    inputs = inputs.loc[:, names].copy()
    for name in names:
        blank = inputs[name].isna() | inputs[name].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(name + ": missing or blank input; all five inputs are required.")

    numeric = {}
    for entry in spec["inputs"]:
        name = entry["name"]
        if name == "calcification":
            continue
        values = pd.to_numeric(inputs[name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(name + ": enter finite numeric values.")
        if "allowed_values" in entry:
            valid = np.isin(values, entry["allowed_values"])
        else:
            valid = (values >= entry["allowed_range"][0]) & (values <= entry["allowed_range"][1])
        if not valid.all():
            allowed = entry.get("allowed_values", entry.get("allowed_range"))
            raise ValueError(name + ": value outside allowed values/range " + str(allowed))
        numeric[name] = values

    grades = inputs["calcification"].astype(str).str.strip().str.lower().to_numpy()
    levels = ("none", "mild", "moderate", "severe")
    if not np.isin(grades, levels).all():
        raise ValueError("calcification: enter none, mild, moderate, or severe.")
    scaling = spec["length_standardization"]
    # Exact fitted feature order. The final indicator is the training-time
    # missing category, always false for these validated, complete inputs.
    # These are structural one-hot zeros, never replacements for missing inputs.
    encoded = np.column_stack([
        (numeric["occlusion_length_mm"] - scaling["mean_mm"]) / scaling["scale_mm"],
        numeric["good_distal_landing_zone"],
        1.0 - numeric["blunt_stump"],
        numeric["bend_gt45"],
        *[grades == level for level in levels],
        grades == "__MISSING__",
    ])
    model = xgb.Booster()
    model.load_model("model_xgb_technical_success.json")
    success = model.predict(xgb.DMatrix(encoded)).astype(float)
    return pd.DataFrame({"probability_technical_success": success,
                         "probability_failure": 1.0 - success})


def main():
    filename = input("CSV filename [example_input.csv]: ").strip() or "example_input.csv"
    try:
        inputs = pd.read_csv(filename, dtype=str, keep_default_na=False,
                             skip_blank_lines=False)
        result = predict(inputs)
    except (ValueError, OSError, xgb.core.XGBoostError) as error:
        raise SystemExit("Input/prediction error: " + str(error)) from None
    print()
    print(result.to_csv(index=False, float_format="%.9f", lineterminator="\n"), end="")


if __name__ == "__main__":
    main()
