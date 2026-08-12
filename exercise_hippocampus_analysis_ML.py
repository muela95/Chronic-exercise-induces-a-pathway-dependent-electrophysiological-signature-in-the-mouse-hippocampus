"""
================================================================================
Pathway-specific classification of exercised vs. sedentary mice from
hippocampal field-potential generators.
================================================================================

Companion code for:
  "Chronic exercise induces a pathway-dependent electrophysiological signature
   in the mouse hippocampus"

The script takes field-potential generators recovered by ICA from laminar
recordings, extracts a set of features from short signal windows, and tests
whether a classifier can tell exercised from sedentary animals using each
generator separately.

GENERATORS
--------------------------------------------------------------------------------
  LM        LM8-todo-nuevo.mat      lacunosum-moleculare
  PL        PL6-todo.mat            lateral perforant path (LPP)
  Schaffer  Schff8-todo-nuevo.mat   Schaffer collateral

USAGE
--------------------------------------------------------------------------------
  python exercise_hippocampus_analysis.py            # analyse all three
  python exercise_hippocampus_analysis.py PL         # just the LPP generator
  python exercise_hippocampus_analysis.py LM Schaffer
(or set GENERATOR_DEFAULT below and run without arguments in an IDE)

ANALYSIS DESIGN
--------------------------------------------------------------------------------
The animal, not the signal window, is the unit of inference. Every step that
depends on the data is confined to the training animals of each fold:

1. Window size and overlap are fixed a priori on physiological grounds
   (WINDOW_SIZE, OVERLAP) and are not selected from the data.
2. Outer folds hold out one control and one exercised animal at a time. All
   windows from those two animals are excluded from training.
3. Within each fold, VIF filtering, standardization and the classifier are fit
   on the training animals only, as a single sklearn Pipeline.
4. Hyperparameters are tuned by an inner, animal-grouped cross-validation that
   sees only that fold's training animals.
5. The permutation test refits the same pipeline in every fold with group
   labels shuffled across animals, so the null and the observed value come from
   the same procedure.
6. Performance is reported at the level of animals, with a Wilson confidence
   interval, and at the level of segments as a secondary measure.

Analyses are restricted to epochs without sustained theta, so no theta-gamma
coupling measure is computed.
================================================================================
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal, stats
from scipy.signal import find_peaks

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

try:
    from numpy import trapezoid as _trapz    # numpy >= 2.0
except ImportError:
    from numpy import trapz as _trapz        # numpy < 2.0

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Data ---------------------------------------------------------------------
# Verify this base directory. Note the accent in "Artículos". On some systems the
# Documents folder is localised ("Documentos"); adjust if needed.
BASE_DIR = r"C:\Users\pmuel\Documents\Laboratorio\Artículos\codigo"

GENERATORS = {
    "LM":       {"file": "LM8-todo-nuevo.mat",   "prefix": "lm",       "label": "lacunosum-moleculare"},
    "PL":       {"file": "PL6-todo.mat",         "prefix": "lpp",      "label": "lateral perforant path"},
    "Schaffer": {"file": "Schff8-todo-nuevo.mat","prefix": "schaffer", "label": "Schaffer collateral"},
}
GENERATOR_DEFAULT = "all"   # used when no command-line argument is given

FS = 2500  # sampling rate (Hz)

# --- Windowing: fixed a priori, not tuned on the data -------------------------
WINDOW_SIZE = 5000            # samples (2.0 s at 2500 Hz)
OVERLAP = 0.5                 # 50%
MIN_SEGMENTS_PER_ANIMAL = 5   # animals with fewer usable windows are excluded

# --- Nested cross-validation --------------------------------------------------
RANDOM_STATE = 42
INNER_SPLITS = 3              # inner animal-level folds for hyperparameter tuning
PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [2, 3],
    "clf__learning_rate": [0.05, 0.1],
    "clf__subsample": [0.8],
}
VIF_THRESHOLD = 5.0

# --- Fixed pipeline used for the permutation test -----------------------------
FIXED_MODEL_PARAMS = {
    "n_estimators": 60,     # lighter than the tuned model. Used only to build
    "max_depth": 3,         # the permutation null, not for the reported
    "learning_rate": 0.1,   # accuracy, so it keeps the test tractable without
    "subsample": 0.8,       # affecting the nested-CV performance.
    # random_state is injected by make_pipeline(); do not set it here.
}
N_PERMUTATIONS = 200          # resolution to about p = 0.005


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(tag="run"):
    fname = f"analysis_{tag}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("exercise_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(fname, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = logging.getLogger("exercise_analysis")


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(file_path):
    """Load control/exercised recordings from the MATLAB struct G.s.
    Layout: G.s is a 2 x N cell array, row 0 = control, row 1 = exercised,
    one column per animal. Empty cells are skipped."""
    logger.info(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{file_path} not found. Check path/extension. If it is a MATLAB "
            f"v7.3 (HDF5) file, load with h5py instead of scipy.io.loadmat."
        )
    data = scipy.io.loadmat(file_path)
    if "G" not in data:
        raise KeyError(f"'G' not found in {file_path}. Keys: {list(data.keys())}")
    g_s = data["G"]["s"][0, 0]
    n_cols = g_s.shape[1]
    logger.info(f"G.s shape: {g_s.shape} (row 0=control, row 1=exercised)")

    def extract_row(row):
        subjects = []
        for i in range(n_cols):
            try:
                raw = np.asarray(g_s[row, i].flatten(), dtype=np.float64)
                raw = raw[np.isfinite(raw)]
                if len(raw) > 0:
                    subjects.append(raw)
            except Exception:
                continue
        return subjects

    control = extract_row(0)
    exercised = extract_row(1)
    logger.info(f"Loaded {len(control)} control and {len(exercised)} exercised animals")
    if len(control) == 0 or len(exercised) == 0:
        raise ValueError("One of the groups is empty; check the G.s layout.")
    return control, exercised


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
# FEATURE EXTRACTION
# =============================================================================
def feature_names():
    names = [
        # time-domain / morphological
        "min", "max", "zero_crossings", "energy", "rms", "signal_variability",
        "diff_variance", "num_peaks", "peak_valley_ratio", "std_peak_height",
        # absolute band power
        "delta_power", "theta_power", "alpha_power", "beta_power",
        "slow_gamma_power", "fast_gamma_power", "high_freq_power",
        # relative band power
        "delta_rel", "theta_rel", "alpha_rel", "beta_rel",
        "slow_gamma_rel", "fast_gamma_rel", "high_freq_rel",
        # bounded contrasts
        "fast_slow_gamma_ratio", "gamma_beta_ratio",
    ]
    return names
 


def extract_features(x, fs=FS):
    """One feature vector from a single window, in canonical feature_names() order."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    f = {}
    f["min"] = np.min(x)
    f["max"] = np.max(x)
    f["zero_crossings"] = np.sum(np.diff(np.signbit(x)))
    f["energy"] = np.sum(x ** 2)
    f["rms"] = np.sqrt(np.mean(x ** 2))
    d = np.diff(x)
    f["signal_variability"] = np.std(d)
    f["diff_variance"] = np.var(d)
    try:
        thr = np.mean(x) + 0.5 * np.std(x)
        peaks, _ = find_peaks(x, height=thr)
        valleys, _ = find_peaks(-x, height=-np.mean(x))
        f["num_peaks"] = len(peaks)
        f["peak_valley_ratio"] = len(peaks) / (len(valleys) + 1)
        f["std_peak_height"] = np.std(x[peaks]) if len(peaks) else 0.0
    except Exception:
        f["num_peaks"] = f["peak_valley_ratio"] = f["std_peak_height"] = 0.0

    try:
    # 1 Hz resolution at fs = 2500 Hz; window is 5000 samples (2 s) -> 3 Welch
    # segments with 50% overlap. Trade-off: coarser averaging, correct bands.
        nperseg = int(min(len(x), fs))
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
        df_res = freqs[1] - freqs[0]
        nyq = fs / 2
 
        bands = {
            "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30),
            "slow_gamma": (30, 60), "fast_gamma": (60, 100),
            "high_freq": (100, min(300, nyq - 1)),
        }
 
        def bp(band):
            """Band power, robust to bands containing 0 or 1 frequency bin."""
            lo, hi = band
            if lo >= nyq:
                return 0.0
            idx = (freqs >= lo) & (freqs <= hi)
            n_bins = int(np.count_nonzero(idx))
            if n_bins == 0:
                return 0.0
            if n_bins == 1:                      # trapezoid over 1 point = 0 -> use rectangle
                return float(psd[idx][0] * df_res)
            return float(_trapz(psd[idx], freqs[idx]))
 
        for name, band in bands.items():
            f[f"{name}_power"] = bp(band)
 
        total = float(_trapz(psd, freqs))
        EPS = 1e-12
 
        # relative powers (bounded, 0-1) -- replace the divide-by-theta ratios
        for name in bands:
            f[f"{name}_rel"] = f[f"{name}_power"] / total if total > 0 else 0.0
 
        # bounded band contrasts (no explosive denominators)
        f["fast_slow_gamma_ratio"] = f["fast_gamma_power"] / (f["slow_gamma_power"] + EPS)
        f["gamma_beta_ratio"] = (f["slow_gamma_power"] + f["fast_gamma_power"]) / (f["beta_power"] + EPS)
 
    except Exception as e:
        logger.warning(f"Spectral analysis failed: {e}")
        for k in [f"{b}_power" for b in ["delta", "theta", "alpha", "beta",
                                     "slow_gamma", "fast_gamma", "high_freq"]] + \
             [f"{b}_rel" for b in ["delta", "theta", "alpha", "beta",
                                   "slow_gamma", "fast_gamma", "high_freq"]] + \
             ["fast_slow_gamma_ratio", "gamma_beta_ratio"]:
            f[k] = 0.0

    vec = [f.get(k, 0.0) for k in feature_names()]
    vec = [0.0 if (v is None or not np.isfinite(v)) else v for v in vec]
    return vec


def build_dataset(control, exercised):
    """Build the windowed feature matrix. Window and overlap are fixed a priori.
    Animals with fewer than MIN_SEGMENTS_PER_ANIMAL usable windows are excluded
    and reported."""
    step = max(1, int(WINDOW_SIZE * (1 - OVERLAP)))
    X, y, groups = [], [], []
    seg_counts, excluded = {}, []

    def process(subjects, label, tag):
        for i, sig in enumerate(subjects):
            aid = f"{tag}_{i}"
            windows = []
            for s in range(0, len(sig) - WINDOW_SIZE + 1, step):
                w = sig[s:s + WINDOW_SIZE]
                if len(w) == WINDOW_SIZE:
                    fv = extract_features(w)
                    if fv is None or len(fv) != len(feature_names()):
                        raise RuntimeError(
                            f"extract_features returned {type(fv).__name__} "
                            f"(expected {len(feature_names())} floats) for {aid}")
                    windows.append(fv)
            seg_counts[aid] = len(windows)
            if len(windows) < MIN_SEGMENTS_PER_ANIMAL:
                excluded.append((aid, len(windows)))
                logger.warning(f"  EXCLUDED {aid}: {len(windows)} windows "
                               f"(< {MIN_SEGMENTS_PER_ANIMAL})")
                continue
            for wv in windows:
                X.append(wv); y.append(label); groups.append(aid)
            logger.info(f"  {aid}: {len(windows)} windows")

    logger.info("Building dataset (control)...")
    process(control, 0, "control")
    logger.info("Building dataset (exercised)...")
    process(exercised, 1, "exercised")

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_ctrl = len({g for g, yy in zip(groups, y) if yy == 0})
    n_exer = len({g for g, yy in zip(groups, y) if yy == 1})
    logger.info(f"Included: {n_ctrl} control + {n_exer} exercised; "
                f"{X.shape[0]} windows; {X.shape[1] if X.ndim==2 else 0} features")
    if excluded:
        logger.info(f"Excluded {len(excluded)} animal(s): {excluded}")
    
    dead = [fn for j, fn in enumerate(feature_names()) if np.std(X[:, j]) < 1e-12]
    if dead:
        logger.warning(f"Zero-variance features (NOT aborting): {dead}")
    return X, y, groups, seg_counts, excluded


# =============================================================================
# LEAK-FREE PIPELINE COMPONENTS
# =============================================================================
class VIFFilter(BaseEstimator, TransformerMixin):
    """Iteratively drop features with VIF above the threshold. Unsupervised, and
    fit on training data only; inside the Pipeline it is refit for each fold."""

    def __init__(self, threshold=5.0, min_features=2):
        self.threshold = threshold
        self.min_features = min_features

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        keep = list(range(X.shape[1]))
        while len(keep) > self.min_features:
            Xk = X[:, keep]
            Xs = (Xk - Xk.mean(0)) / (Xk.std(0) + 1e-12)
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    vifs = [variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])]
            except Exception:
                break
            vifs = np.asarray(vifs, dtype=float)
            vifs[~np.isfinite(vifs)] = np.inf
            j = int(np.argmax(vifs))
            if vifs[j] <= self.threshold:
                break
            keep.pop(j)
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.keep_]


def make_pipeline(model_params=None):
    clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **(model_params or {}))
    return Pipeline([
        ("vif", VIFFilter(threshold=VIF_THRESHOLD)),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# =============================================================================
# OUTER CROSS-VALIDATION: leave-one-control-one-exercised-out
# =============================================================================
def outer_folds(groups, y):
    animal_label = {g: yy for g, yy in zip(groups, y)}
    ctrl = sorted([g for g, l in animal_label.items() if l == 0])
    exer = sorted([g for g, l in animal_label.items() if l == 1])
    for ca, ea in product(ctrl, exer):
        test_mask = (groups == ca) | (groups == ea)
        yield np.where(~test_mask)[0], np.where(test_mask)[0], ca, ea


def t_ci(vals, confidence=0.95):
    """Two-sided t CI, always returns (low, high) with low<=high."""
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float(np.mean(vals)), float(np.mean(vals)))
    m = np.mean(vals); se = stats.sem(vals)
    h = se * stats.t.ppf((1 + confidence) / 2.0, len(vals) - 1)
    return (float(min(m - h, m + h)), float(max(m - h, m + h)))


# =============================================================================
# NESTED CROSS-VALIDATION: tuning happens inside each outer fold
# =============================================================================
def nested_cv(X, y, groups):
    logger.info("=" * 70)
    logger.info("NESTED CROSS-VALIDATION (tuning inside each fold)")
    logger.info("=" * 70)

    outer_acc, outer_auc = [], []
    per_animal_prob, per_animal_true = {}, {}
    fold_importances = []
    fnames = feature_names()

    fold_records = []          # one row per outer fold (for figures/tables)
    fold_imp_long = []         # (fold, feature, importance) long format
    segment_predictions = []   # (fold, animal, group, true_label, pred_prob) for ROC etc.

    fold_list = list(outer_folds(groups, y))
    logger.info(f"Outer folds (control x exercised pairs): {len(fold_list)}")

    for k, (tr, te, ca, ea) in enumerate(fold_list):
        Xtr, Xte, ytr, yte, gtr = X[tr], X[te], y[tr], y[te], groups[tr]

        n_ctrl_tr = len({g for g, l in zip(gtr, ytr) if l == 0})
        n_exer_tr = len({g for g, l in zip(gtr, ytr) if l == 1})
        n_splits = max(2, min(INNER_SPLITS, n_ctrl_tr, n_exer_tr))
        inner_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=RANDOM_STATE)

        grid = GridSearchCV(estimator=make_pipeline(), param_grid=PARAM_GRID,
                            cv=inner_cv, scoring="accuracy", n_jobs=-1, refit=True)
        grid.fit(Xtr, ytr, groups=gtr)
        best = grid.best_estimator_

        yte_pred = best.predict(Xte)
        yte_prob = best.predict_proba(Xte)[:, 1]
        fold_acc = accuracy_score(yte, yte_pred)
        outer_acc.append(fold_acc)
        try:
            fold_auc = roc_auc_score(yte, yte_prob)
            outer_auc.append(fold_auc)
        except Exception:
            fold_auc = float("nan")

        for aid in (ca, ea):
            mask = groups[te] == aid
            per_animal_prob.setdefault(aid, []).extend(yte_prob[mask].tolist())
            per_animal_true[aid] = int(y[groups == aid][0])

        vif = best.named_steps["vif"]
        imp = best.named_steps["clf"].feature_importances_
        imp_map = {fnames[idx]: float(val) for idx, val in zip(vif.keep_, imp)}
        fold_importances.append(imp_map)

        # --- per-fold records kept for figures and tables --------------------
        # per-fold summary
        prob_ca = float(np.mean(yte_prob[groups[te] == ca]))
        prob_ea = float(np.mean(yte_prob[groups[te] == ea]))
        rec = {"fold": k + 1, "test_control_animal": ca, "test_exercised_animal": ea,
               "n_train_segments": int(len(tr)), "n_test_segments": int(len(te)),
               "accuracy": float(fold_acc), "auc": float(fold_auc),
               "mean_prob_control_animal": prob_ca,
               "mean_prob_exercised_animal": prob_ea,
               "n_features_after_vif": int(len(vif.keep_))}
        rec.update({k2: v2 for k2, v2 in grid.best_params_.items()})  # best hyperparams
        fold_records.append(rec)
        # per-fold feature importances (long); features dropped by VIF -> NaN
        for fn in fnames:
            fold_imp_long.append({"fold": k + 1, "feature": fn,
                                  "importance": imp_map.get(fn, np.nan)})
        # segment-level test predictions
        for aid in (ca, ea):
            m = groups[te] == aid
            grp = "control" if aid.startswith("control") else "exercised"
            for prob in yte_prob[m]:
                segment_predictions.append({"fold": k + 1, "animal": aid, "group": grp,
                                            "true_label": int(y[groups == aid][0]),
                                            "pred_prob_exercised": float(prob)})

        logger.info(f"Fold {k+1}/{len(fold_list)} [test: {ca}, {ea}] "
                    f"acc={fold_acc:.3f} auc={fold_auc:.3f} "
                    f"n_feat={len(vif.keep_)} best={grid.best_params_}")

    seg_mean = float(np.mean(outer_acc))
    seg_ci = t_ci(outer_acc)
    auc_mean = float(np.mean(outer_auc)) if outer_auc else float("nan")

    rows = []
    for aid, probs in per_animal_prob.items():
        mp = float(np.mean(probs)); pred = int(mp > 0.5)
        rows.append({"animal": aid, "true_label": per_animal_true[aid],
                     "mean_pred_prob_exercised": mp, "predicted_label": pred,
                     "correct": int(pred == per_animal_true[aid]),
                     "n_prediction_instances": len(probs)})
    animal_df = pd.DataFrame(rows).sort_values("animal")
    n_correct = int(animal_df["correct"].sum())
    n_animals = len(animal_df)
    animal_acc = n_correct / n_animals if n_animals else float("nan")
    # Wilson CI for the animal-level proportion, the primary metric
    if n_animals > 0:
        a_lo, a_hi = proportion_confint(n_correct, n_animals, alpha=0.05, method="wilson")
    else:
        a_lo, a_hi = float("nan"), float("nan")

    imp_rows = []
    for fn in fnames:
        vals = [d.get(fn, 0.0) for d in fold_importances]
        retained = np.mean([fn in d for d in fold_importances]) if fold_importances else 0.0
        imp_rows.append({"feature": fn, "mean_importance": float(np.mean(vals)),
                         "retention_rate": float(retained)})
    importance_df = pd.DataFrame(imp_rows).sort_values("mean_importance", ascending=False)

    logger.info("-" * 70)
    logger.info(f"Segment-level accuracy: {seg_mean:.4f}  "
                f"(fold CI, descriptive) [{seg_ci[0]:.4f}, {seg_ci[1]:.4f}]")
    logger.info(f"Segment-level AUC:      {auc_mean:.4f}")
    logger.info(f"ANIMAL-level accuracy:  {animal_acc:.4f}  "
                f"({n_correct}/{n_animals})  Wilson 95% CI [{a_lo:.4f}, {a_hi:.4f}]")
    logger.info("-" * 70)

    return {
        "segment_accuracy_mean": seg_mean,
        "segment_accuracy_fold_ci_descriptive": seg_ci,
        "segment_auc_mean": auc_mean,
        "animal_accuracy": animal_acc,
        "animal_accuracy_wilson_ci": (float(a_lo), float(a_hi)),
        "n_animals_correct": n_correct,
        "n_animals_total": n_animals,
        "outer_fold_accuracies": [float(a) for a in outer_acc],
        "animal_predictions": animal_df,
        "feature_importance": importance_df,
        "fold_results": pd.DataFrame(fold_records),
        "fold_feature_importances_long": pd.DataFrame(fold_imp_long),
        "segment_predictions": pd.DataFrame(segment_predictions),
    }


# =============================================================================
# PERMUTATION TEST (fixed pipeline, animal-level shuffling)
# =============================================================================
def evaluate_fixed_pipeline(X, y, groups):
    accs = []
    for tr, te, _, _ in outer_folds(groups, y):
        pipe = make_pipeline(model_params=FIXED_MODEL_PARAMS)
        pipe.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], pipe.predict(X[te])))
    return float(np.mean(accs))


def permutation_test(X, y, groups, n_permutations=N_PERMUTATIONS):
    logger.info("=" * 70)
    logger.info(f"PERMUTATION TEST (fixed pipeline, animal-level shuffling, "
                f"{n_permutations} permutations)")
    logger.info("=" * 70)

    animal_label = {g: yy for g, yy in zip(groups, y)}
    animals = list(animal_label.keys())
    true_labels = np.array([animal_label[a] for a in animals])

    observed = evaluate_fixed_pipeline(X, y, groups)
    logger.info(f"Observed fixed-pipeline accuracy: {observed:.4f}")

    n_folds = len(list(outer_folds(groups, y)))
    if n_folds * n_permutations > 20000:
        logger.warning(f"~{n_folds * n_permutations} model fits; may be slow. "
                       f"Lower N_PERMUTATIONS for a first pass.")

    rng = np.random.default_rng(RANDOM_STATE)
    null = []
    for i in range(n_permutations):
        amap = dict(zip(animals, rng.permutation(true_labels)))
        y_perm = np.array([amap[g] for g in groups])
        null.append(evaluate_fixed_pipeline(X, y_perm, groups))
        if (i + 1) % 50 == 0:
            logger.info(f"  permutation {i+1}/{n_permutations}")

    null = np.asarray(null)
    p_value = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    logger.info(f"Null: {null.mean():.4f} ± {null.std():.4f}")
    logger.info(f"Permutation p-value: {p_value:.4f} "
                f"({'significant' if p_value < 0.05 else 'n.s.'})")
    return {"observed_accuracy": observed, "p_value": float(p_value),
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_distribution": [float(v) for v in null]}


# =============================================================================
# SUPPLEMENTARY PER-FEATURE TABLE (descriptive; animal-level replicate)
# =============================================================================
def feature_table(X, y, groups, importance_df):
    fnames = feature_names()
    ctrl, exer = [], []
    for aid in np.unique(groups):
        mask = groups == aid
        (ctrl if y[mask][0] == 0 else exer).append(X[mask].mean(axis=0))
    ctrl, exer = np.asarray(ctrl), np.asarray(exer)

    rows, pvals = [], []
    for j, fn in enumerate(fnames):
        c, e = ctrl[:, j], exer[:, j]
        denom = (len(c) + len(e) - 2)
        pooled = np.sqrt(((len(c) - 1) * np.var(c, ddof=1) +
                          (len(e) - 1) * np.var(e, ddof=1)) / denom) if denom > 0 else np.nan
        d = (np.mean(e) - np.mean(c)) / pooled if (pooled and pooled > 0) else np.nan
        try:
            u, p = stats.mannwhitneyu(e, c, alternative="two-sided")
        except Exception:
            u, p = np.nan, np.nan
        pvals.append(p)
        rows.append({"feature": fn,
                     "control_mean": float(np.mean(c)), "control_sd": float(np.std(c, ddof=1)),
                     "exercised_mean": float(np.mean(e)), "exercised_sd": float(np.std(e, ddof=1)),
                     "cohens_d": float(d) if np.isfinite(d) else np.nan,
                     "mannwhitney_U": float(u) if np.isfinite(u) else np.nan,
                     "p_raw": float(p) if np.isfinite(p) else np.nan})
    pvals = np.array(pvals, dtype=float)
    valid = np.isfinite(pvals)
    p_adj = np.full(len(pvals), np.nan)
    if valid.sum() > 0:
        p_adj[valid] = multipletests(pvals[valid], method="fdr_bh")[1]
    for r, pa in zip(rows, p_adj):
        r["p_bh_corrected"] = float(pa) if np.isfinite(pa) else np.nan

    df = pd.DataFrame(rows).merge(importance_df[["feature", "mean_importance"]],
                                  on="feature", how="left")
    return df.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)


# =============================================================================
# PER-GENERATOR ORCHESTRATION
# =============================================================================
def analyze(X, y, groups, seg_counts, excluded, prefix, label,
            out_dir=".", n_permutations=N_PERMUTATIONS):
    """Run the analysis for one already-built dataset and write the output files."""
    if X.ndim != 2 or len(np.unique(y)) < 2:
        logger.error(f"[{label}] insufficient data / only one class; skipping.")
        return None

    nested = nested_cv(X, y, groups)
    perm = permutation_test(X, y, groups, n_permutations=n_permutations)
    feat_df = feature_table(X, y, groups, nested["feature_importance"])

    def path(suffix):
        return os.path.join(out_dir, f"{prefix}_{suffix}")

    nested["animal_predictions"].to_csv(path("animal_level_predictions.csv"), index=False)
    nested["feature_importance"].to_csv(path("feature_importance.csv"), index=False)
    feat_df.to_csv(path("feature_supplementary_table.csv"), index=False)

    # --- per-fold outputs, kept for figures and tables -------------------------
    nested["fold_results"].to_csv(path("fold_results.csv"), index=False)
    nested["fold_feature_importances_long"].to_csv(path("fold_feature_importances.csv"), index=False)
    nested["segment_predictions"].to_csv(path("segment_predictions.csv"), index=False)
    # permutation null distribution (for the permutation histogram)
    pd.DataFrame({"permutation": range(1, len(perm["null_distribution"]) + 1),
                  "null_accuracy": perm["null_distribution"]}).to_csv(
        path("permutation_null_distribution.csv"), index=False)

    # --- VIF on the full dataset. Descriptive only: the selection that the
    #     model uses happens per fold, inside the pipeline. --------------------
    try:
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            vifs = [variance_inflation_factor(Xs, i) for i in range(Xs.shape[1])]
        pd.DataFrame({"feature": feature_names(),
                      "VIF_full_dataset": [float(v) for v in vifs]}).to_csv(
            path("vif_diagnostic_full_dataset.csv"), index=False)
    except Exception as e:
        logger.warning(f"VIF diagnostic failed: {e}")

    seg_df = pd.DataFrame(
        [{"animal": a, "n_segments": n,
          "group": ("control" if a.startswith("control") else "exercised"),
          "included": a not in [e[0] for e in excluded]}
         for a, n in seg_counts.items()]
    ).sort_values(["group", "animal"])
    seg_df.to_csv(path("segments_per_animal.csv"), index=False)

    # Environment information, for reproducibility
    try:
        import sklearn, scipy, statsmodels, platform
        versions = {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__,
                    "scikit_learn": sklearn.__version__,
                    "statsmodels": statsmodels.__version__}
    except Exception:
        versions = {}

    summary = {
        "generator": label, "prefix": prefix,
        "config": {"window_size": WINDOW_SIZE, "overlap": OVERLAP, "fs": FS,
                   "vif_threshold": VIF_THRESHOLD,
                   "param_grid": PARAM_GRID, "inner_splits": INNER_SPLITS,
                   "fixed_model_params_for_permutation": FIXED_MODEL_PARAMS,
                   "min_segments_per_animal": MIN_SEGMENTS_PER_ANIMAL,
                   "random_state": RANDOM_STATE, "n_permutations": n_permutations},
        "software_versions": versions,
        "n_animals": {
            "control_included": int(seg_df[(seg_df.group == "control") & seg_df.included].shape[0]),
            "exercised_included": int(seg_df[(seg_df.group == "exercised") & seg_df.included].shape[0]),
            "excluded": excluded},
        "n_windows": int(X.shape[0]), "n_features": int(X.shape[1]),
        "nested_cv": {k: nested[k] for k in
                      ["segment_accuracy_mean", "segment_accuracy_fold_ci_descriptive",
                       "segment_auc_mean", "animal_accuracy", "animal_accuracy_wilson_ci",
                       "n_animals_correct", "n_animals_total", "outer_fold_accuracies"]},
        "permutation_test": perm,
    }
    with open(path("summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"[{label}] files written with prefix '{prefix}_':")
    for suf in ["animal_level_predictions.csv", "feature_importance.csv",
                "feature_supplementary_table.csv", "fold_results.csv",
                "fold_feature_importances.csv", "segment_predictions.csv",
                "permutation_null_distribution.csv", "vif_diagnostic_full_dataset.csv",
                "segments_per_animal.csv", "summary.json"]:
        logger.info(f"    {prefix}_{suf}")
    return summary


def run_generator(key, out_dir="."):
    spec = GENERATORS[key]
    file_path = os.path.join(BASE_DIR, spec["file"])
    logger.info("#" * 70)
    logger.info(f"GENERATOR: {spec['label']}  ({spec['file']})")
    logger.info("#" * 70)
    control, exercised = load_data(file_path)
    X, y, groups, seg_counts, excluded = build_dataset(control, exercised)
    return analyze(X, y, groups, seg_counts, excluded,
                   prefix=spec["prefix"], label=spec["label"], out_dir=out_dir)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        selection = ["LM", "PL", "Schaffer"] if GENERATOR_DEFAULT == "all" else [GENERATOR_DEFAULT]
    elif len(argv) == 1 and argv[0].lower() == "all":
        selection = ["LM", "PL", "Schaffer"]
    else:
        selection = []
        for a in argv:
            match = next((k for k in GENERATORS if k.lower() == a.lower()), None)
            if match is None:
                raise SystemExit(f"Unknown generator '{a}'. Options: {list(GENERATORS)} or 'all'.")
            selection.append(match)

    global logger
    logger = setup_logging(tag="_".join(selection))
    # An infinite VIF marks a redundant feature that will be dropped, so the
    # divide-by-zero warnings it raises are expected.
    np.seterr(divide="ignore", invalid="ignore")
    logger.info(f"Config: window={WINDOW_SIZE} ({WINDOW_SIZE/FS:.2f}s), overlap={OVERLAP}")
    logger.info(f"Analysing generators: {selection}")

    results = {}
    for key in selection:
        try:
            results[key] = run_generator(key)
        except Exception as e:
            logger.exception(f"Generator {key} failed: {e}")   # exception, no error
            results[key] = None
            
    # Combined comparison across generators (the key cross-pathway result)
    logger.info("=" * 70)
    logger.info("CROSS-GENERATOR COMPARISON")
    logger.info("=" * 70)
    comp_rows = []
    for key, r in results.items():
        if r is None:
            comp_rows.append({"generator": GENERATORS[key]["label"], "status": "failed"})
            continue
        n = r["nested_cv"]; p = r["permutation_test"]
        comp_rows.append({
            "generator": r["generator"],
            "animal_accuracy": round(n["animal_accuracy"], 4),
            "animal_wilson_ci": [round(c, 4) for c in n["animal_accuracy_wilson_ci"]],
            "segment_accuracy": round(n["segment_accuracy_mean"], 4),
            "segment_auc": round(n["segment_auc_mean"], 4),
            "perm_p_value": round(p["p_value"], 4),
            "n_control": r["n_animals"]["control_included"],
            "n_exercised": r["n_animals"]["exercised_included"],
        })
    comp_df = pd.DataFrame(comp_rows)
    logger.info("\n" + comp_df.to_string(index=False))
    comp_df.to_csv("cross_generator_comparison.csv", index=False)
    with open("cross_generator_comparison.json", "w") as f:
        json.dump(comp_rows, f, indent=2, default=str)
    logger.info("Wrote cross_generator_comparison.{csv,json}")


if __name__ == "__main__":
    main()