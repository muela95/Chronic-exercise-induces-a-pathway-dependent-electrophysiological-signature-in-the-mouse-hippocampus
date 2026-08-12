# Chronic exercise induces a pathway-dependent electrophysiological signature in the mouse hippocampus

Code accompanying the manuscript. It classifies exercised versus sedentary mice
from hippocampal field-potential generators recovered by independent component
analysis of laminar recordings.

## Contents

| File | Description |
|---|---|
| `exercise_hippocampus_analysis.py` | Feature extraction, nested cross-validation, permutation testing and per-feature statistics |
| `concatenar_mats_pl.py`, `concatenar_mats_LM.py`, `concatenar_mats_Schaffer.py` | Collect the recordings of a given generator into one file per group (sedentary, exercised) |
| `union6_pl.m`, `union7.m`, `union8.m` | Merge the two group files of each generator into the single `.mat` read by the analysis script |

The two assembly steps are run once per generator, before the analysis:
the Python scripts group the recordings by condition and the MATLAB scripts
merge them into `PL6-todo.mat`, `LM8-todo-nuevo.mat` and `Schff8-todo-nuevo.mat`.

## Requirements

Python 3.11 with numpy, pandas, scipy, scikit-learn (1.6.1) and statsmodels
(0.14.4). The exact versions used are written to `<prefix>_summary.json` on
every run.

## Usage

Set `BASE_DIR` at the top of the script to the folder containing the `.mat`
files, then run:

```
python exercise_hippocampus_analysis.py            # all three generators
python exercise_hippocampus_analysis.py PL         # lateral perforant path only
python exercise_hippocampus_analysis.py LM Schaffer
```

## Analysis

The animal is the unit of inference. Signal windows are 2 s with 50% overlap,
fixed a priori rather than selected from the data, and 26 features are extracted
from each window.

Outer folds hold out one control and one exercised animal at a time, so all
windows from those two animals are excluded from training. Within each fold, VIF
filtering, standardization and the gradient boosting classifier are fit on the
training animals only, as a single sklearn Pipeline, and hyperparameters are
tuned by an inner animal-grouped cross-validation that sees only that fold's
training animals. The permutation test refits the same pipeline in every fold
with labels shuffled across animals.

Performance is reported at the level of animals, with a Wilson confidence
interval, and at the level of segments as a secondary measure.

Analyses are restricted to epochs without sustained theta, so no theta-gamma
coupling measure is computed.

## Output

For each generator, with prefix `lpp`, `lm` or `schaffer`:

| File | Contents |
|---|---|
| `<prefix>_summary.json` | Configuration, software versions, animal counts and all performance metrics |
| `<prefix>_animal_level_predictions.csv` | Mean predicted probability and call for each animal |
| `<prefix>_fold_results.csv` | Accuracy, AUC, retained features and selected hyperparameters per fold |
| `<prefix>_segment_predictions.csv` | Window-level predicted probabilities |
| `<prefix>_feature_importance.csv` | Mean importance and retention rate per feature |
| `<prefix>_fold_feature_importances.csv` | Importances per fold |
| `<prefix>_feature_supplementary_table.csv` | Per-feature group comparison on animal-level means |
| `<prefix>_permutation_null_distribution.csv` | Null accuracies |
| `<prefix>_vif_diagnostic_full_dataset.csv` | VIF on the full dataset, descriptive only |
| `<prefix>_segments_per_animal.csv` | Windows contributed by each animal |

A `cross_generator_comparison.csv` and `.json` summarise the three generators.

## Data

The raw recordings are deposited at
[https://zenodo.org/records/20281330](https://zenodo.org/records/20281330).
The assembly scripts in this repository take those recordings and produce the
per-generator `.mat` files that the analysis script reads.
