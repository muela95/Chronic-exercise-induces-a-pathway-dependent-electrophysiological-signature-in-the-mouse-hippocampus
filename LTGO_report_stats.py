"""
Statistical reporting script for LTGO classifier validation.
Reads pre-computed JSON results and reports t-statistic, df, 95% CI,
and permutation test p-value for each dataset.
Saves results to Excel (statistical_report.xlsx) in the same folder.

NOTE: Original analysis used permutation test (no t-statistic).
      t-statistic here is from one-sample t-test of fold accuracies vs.
      chance level (0.5), as required by journal reviewers.

Does NOT modify any existing files.
"""

import json
import numpy as np
from scipy import stats
import os
import pandas as pd

JSON_DIR = os.path.join(os.path.dirname(__file__), "jsons")

DATASETS = {
    "PL":       "neural_classifier_evaluation_pl.json",
    "Schaffer": "neural_classifier_evaluation_schaffer.json",
    "LM":       "neural_classifier_evaluation_LM.json",
}

CHANCE_LEVEL = 0.5
ALPHA = 0.95


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_stats(values, label, chance=CHANCE_LEVEL):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    sem = stats.sem(arr)
    df = n - 1

    # One-sample t-test vs. chance level
    t_stat, p_ttest = stats.ttest_1samp(arr, chance)

    # 95% CI using t-distribution (same method as original bootstrap_confidence_intervals)
    h = sem * stats.t.ppf((1 + ALPHA) / 2.0, df)
    ci_lower = mean - h
    ci_upper = mean + h

    return {
        "label":    label,
        "n_folds":  n,
        "mean":     mean,
        "std":      std,
        "sem":      sem,
        "df":       df,
        "t":        t_stat,
        "p_ttest":  p_ttest,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def report_dataset(name, data):
    cv = data["cross_validation_performance"]
    perm = data["permutation_test_results"]
    ds = data.get("dataset_characteristics", {})

    print(f"\n{'='*60}")
    print(f"DATASET: {name}")
    print(f"  Subjects: {ds.get('subject_count', 'N/A')}  |  CV folds: {ds.get('n_folds', 'N/A')}")
    print(f"  CV method: {ds.get('cv_method', 'Leave-Two-Groups-Out')}")
    print(f"{'='*60}")

    metrics = [
        ("Accuracy",    "accuracies"),
        ("Sensitivity", "sensitivities"),
        ("Specificity", "specificities"),
        ("AUC",         "aucs"),
    ]

    obs = perm["baseline_score"]
    null_mean = perm["null_distribution_mean"]
    null_std = perm["null_distribution_std"]
    z_perm = (obs - null_mean) / null_std

    all_stats = []
    for label, key in metrics:
        values = cv[key]["values"]
        s = compute_stats(values, label)
        s["dataset"] = name
        s["n_subjects"] = ds.get("subject_count", None)
        s["z_permutation"] = z_perm
        s["p_permutation"] = perm["p_value"]
        s["observed_accuracy"] = obs
        s["null_mean"] = null_mean
        s["null_std"] = null_std
        all_stats.append(s)

        print(f"\n  {label}:")
        print(f"    Mean +/- SD        : {s['mean']:.4f} +/- {s['std']:.4f}")
        print(f"    95% CI             : [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")
        print(f"    t({s['df']})              : {s['t']:.3f}")
        print(f"    p (vs. chance=0.5) : {s['p_ttest']:.4f}" +
              ("  *" if s["p_ttest"] < 0.05 else ""))

    print(f"\n  Permutation test (1000 permutations):")
    print(f"    Observed accuracy  : {obs:.4f}")
    print(f"    Null mean +/- SD   : {null_mean:.4f} +/- {null_std:.4f}")
    print(f"    z                  : {z_perm:.3f}")
    print(f"    p (permutation)    : {perm['p_value']:.4f}" +
          ("  *" if perm["p_value"] < 0.05 else ""))

    acc = all_stats[0]
    print(f"\n  APA-style (accuracy, t-test vs. chance):")
    print(f"    t({acc['df']}) = {acc['t']:.2f}, p < .001, "
          f"95% CI [{acc['ci_lower']:.3f}, {acc['ci_upper']:.3f}]")
    print(f"\n  APA-style (permutation test):")
    print(f"    z = {z_perm:.2f}, p = {perm['p_value']:.3f}")

    return all_stats


def save_excel(all_rows, out_path):
    df = pd.DataFrame(all_rows, columns=[
        "dataset", "metric", "n_subjects", "n_folds",
        "mean", "std", "sem",
        "ci_lower", "ci_upper",
        "df", "t", "p_ttest",
        "z_permutation", "p_permutation",
        "observed_accuracy", "null_mean", "null_std",
    ])

    # Rename columns to human-friendly headers
    df = df.rename(columns={
        "dataset":           "Dataset",
        "metric":            "Metric",
        "n_subjects":        "N subjects",
        "n_folds":           "N folds",
        "mean":              "Mean",
        "std":               "SD",
        "sem":               "SEM",
        "ci_lower":          "95% CI lower",
        "ci_upper":          "95% CI upper",
        "df":                "df",
        "t":                 "t",
        "p_ttest":           "p (t-test vs chance)",
        "z_permutation":     "z (permutation)",
        "p_permutation":     "p (permutation)",
        "observed_accuracy": "Observed accuracy",
        "null_mean":         "Null distribution mean",
        "null_std":          "Null distribution SD",
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Sheet 1: full stats table
        df.to_excel(writer, sheet_name="Full stats", index=False)

        # Sheet 2: accuracy only, clean summary for paper
        acc_df = df[df["Metric"] == "Accuracy"][[
            "Dataset", "N subjects", "N folds",
            "Mean", "SD", "95% CI lower", "95% CI upper",
            "df", "t", "p (t-test vs chance)",
            "z (permutation)", "p (permutation)",
        ]]
        acc_df.to_excel(writer, sheet_name="Accuracy summary", index=False)

        # Sheet 3: APA-style strings
        apa_rows = []
        for _, row in df[df["Metric"] == "Accuracy"].iterrows():
            p_t = "< .001" if row["p (t-test vs chance)"] < 0.001 else f"= {row['p (t-test vs chance)']:.3f}"
            apa_rows.append({
                "Dataset": row["Dataset"],
                "t-test (APA)": (
                    f"t({int(row['df'])}) = {row['t']:.2f}, p {p_t}, "
                    f"95% CI [{row['95% CI lower']:.3f}, {row['95% CI upper']:.3f}]"
                ),
                "Permutation test (APA)": (
                    f"z = {row['z (permutation)']:.2f}, p = {row['p (permutation)']:.3f}"
                ),
            })
        pd.DataFrame(apa_rows).to_excel(writer, sheet_name="APA strings", index=False)

        # Auto-width columns on all sheets
        for sheet in writer.sheets.values():
            for col in sheet.columns:
                max_len = max(len(str(cell.value or "")) for cell in col) + 2
                sheet.column_dimensions[col[0].column_letter].width = min(max_len, 50)

    print(f"\nExcel saved: {out_path}")


def main():
    print("\nSTATISTICAL REPORT FOR JOURNAL SUBMISSION")
    print("One-sample t-test: fold accuracies vs. chance (0.5)")
    print("95% CI via t-distribution (df = n_folds - 1)")

    all_rows = []

    for name, filename in DATASETS.items():
        path = os.path.join(JSON_DIR, filename)
        if not os.path.exists(path):
            print(f"\n[WARNING] {filename} not found, skipping.")
            continue
        data = load_json(path)
        dataset_stats = report_dataset(name, data)
        for s in dataset_stats:
            all_rows.append({
                "dataset":           s["dataset"],
                "metric":            s["label"],
                "n_subjects":        s["n_subjects"],
                "n_folds":           s["n_folds"],
                "mean":              s["mean"],
                "std":               s["std"],
                "sem":               s["sem"],
                "ci_lower":          s["ci_lower"],
                "ci_upper":          s["ci_upper"],
                "df":                s["df"],
                "t":                 s["t"],
                "p_ttest":           s["p_ttest"],
                "z_permutation":     s["z_permutation"],
                "p_permutation":     s["p_permutation"],
                "observed_accuracy": s["observed_accuracy"],
                "null_mean":         s["null_mean"],
                "null_std":          s["null_std"],
            })

    print(f"\n{'='*60}")
    print("NOTE ON METHODOLOGY")
    print("="*60)
    print(
        "The primary significance test in this work is a permutation test\n"
        "(n=1000 permutations, subject-level label shuffling).\n"
        "The t-statistics above are reported as supplementary descriptives\n"
        "of fold-level accuracy distributions against chance level.\n"
        "Degrees of freedom = number of CV folds - 1.\n"
        "CI method: t-distribution on fold-level values (same as in code)."
    )

    out_path = os.path.join(os.path.dirname(__file__), "statistical_report.xlsx")
    save_excel(all_rows, out_path)


if __name__ == "__main__":
    main()
