# PROGRESS-CTO technical-success prediction model

This package reproduces an individual prediction from the manuscript's fitted
five-predictor XGBoost model. It contains the trained model, summary preprocessing
values, generic prediction code, and one invented example. It contains no registry
records, held-out predictions, training notebook, or data-cleaning scripts.

The positive class is technical success: successful CTO recanalization with
less than 30% residual angiographic stenosis and TIMI grade 3 antegrade flow.
The fitted positive-class weight was 0.8956724014884008 (approximately 0.896),
with random seed 42 and 1,500 trees. The model has not been refitted for release.
**The output was not recalibrated.**

## Inputs

Supply exactly these five columns. Column order in the CSV may vary; the listed
order follows the fitted pipeline's five-predictor input order.

| Column | Definition and allowed input |
| --- | --- |
| `blunt_stump` | Proximal CTO stump morphology: 1 = blunt or no stump; 0 = tapered stump. |
| `occlusion_length_mm` | Angiographic length of the occluded segment, in millimeters; a finite number from 1 through 200. This is a release input limit, not a reported training-data range. |
| `bend_gt45` | 1 = a bend greater than 45 degrees within the CTO segment; 0 = absent. This is not the separate proximal-vessel tortuosity classification. |
| `calcification` | Angiographic grade: `none` = no calcification; `mild` = spot calcifications; `moderate` = involving at most 50% of the reference lesion diameter; `severe` = involving more than 50%. |
| `good_distal_landing_zone` | 1 = a distal vessel segment immediately beyond the CTO at least 2.0 mm in diameter and free of diffuse disease; 0 = otherwise. |

Definitions follow the study manuscript. CTO denotes TIMI grade 0 flow with an
estimated duration of at least three months. All five inputs are required.
Missing values, blank cells, unrecognized grades, nonfinite numbers, extra or
missing columns, and values outside the specified limits produce an error.
No missing input is imputed or replaced with zero.

## Run

Use the Python version recorded in `requirements.txt`. From this directory:

```sh
python -m pip install -r requirements.txt
python predict.py
```

At the CSV filename prompt, press Enter to run `example_input.csv`, or enter the
filename of a CSV with exactly the five input columns. Results print as CSV with
one output row per input row, in the same order. Probabilities are fractions
between 0 and 1; failure probability is exactly 1 minus success probability before
display rounding. Run from this directory so the model and specification can be
found using relative filenames.

For programmatic use from this directory:

```python
import pandas as pd
from predict import predict
print(predict(pd.read_csv("example_input.csv", keep_default_na=False)))
```

The script imports only XGBoost, NumPy, pandas, and Python's built-in `json`.
The requirements also pin their installed runtime dependencies.

## Invented worked example

The example is invented, not selected from the registry: blunt stump = 1,
length = 25 mm, bend = 0, calcification = moderate, good distal landing zone = 1.
Its predicted technical-success probability is **0.864567578** (86.46%), and
failure probability is **0.135432422** (13.54%). The full printed result is in
`example_output.txt`.

## Preprocessing and interpretation

`preprocessing.json` was generated from the fitted training-set preprocessors.
Length is standardized using their exact mean and scale. The training length
imputation mean is 31.598853868194844 mm (31.6 mm rounded); training binary modes
are documented in clinical coding. Those imputation values are documentation
only and are never used to fill a missing prediction input.

The exact nine-column encoding is implemented in `predict.py`. The clinical
blunt-stump indicator is inverted for the model's internal stump feature.
Calcification uses the original four recorded-grade indicators and a fifth
missing-category indicator. With a valid recorded grade, that last indicator is
always zero. These structural one-hot zeros do not substitute for missing inputs.
The one-hot column order is intentionally omitted from the JSON specification.

The intended use is to support procedural planning and informed consent, not to
deny CTO PCI. A prediction does not determine treatment suitability or establish
the expected benefit of PCI compared with other treatments. Interpret it with
clinical assessment and the study's reported validation limitations.

## License and citation

**License pending selection by the model owner.** No open-source license or
additional reuse permission is granted by this draft. The owner must select and
include the license before public release.

Cite the accompanying PROGRESS-CTO five-predictor XGBoost technical-success
manuscript: **DOI: [to be assigned]**. Replace this placeholder with the final
publication citation when available.
