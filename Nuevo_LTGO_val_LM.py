import numpy as np
import scipy.io
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy import stats
from scipy.signal import find_peaks, hilbert
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import logging
import os
import json
from collections import Counter, defaultdict
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_random_state
import psutil
from datetime import datetime
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import hashlib
import pickle
from joblib import Parallel, delayed
from itertools import combinations
warnings.filterwarnings('ignore')

def setup_logging():
    """Initialize comprehensive logging system for model evaluation"""
    log_filename = f"neural_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

class BalancedLeaveTwoGroupsOut(_BaseKFold):
    """
    Custom cross-validator that ensures exactly 2 subjects from each class
    are left out for testing in each fold (4 total test subjects per fold).
    """
    def __init__(self):
        pass
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Calculate number of CV folds"""
        if groups is None or y is None:
            raise ValueError("groups and y must be provided")
        
        unique_groups = np.unique(groups)
        group_labels = {}
        for group in unique_groups:
            group_mask = groups == group
            group_labels[group] = y[group_mask][0]
        
        # Count subjects per class
        class_0_subjects = [g for g, label in group_labels.items() if label == 0]
        class_1_subjects = [g for g, label in group_labels.items() if label == 1]
        
        # Number of ways to choose 2 from each class
        from math import comb
        n_folds = comb(len(class_0_subjects), 2) * comb(len(class_1_subjects), 2)
        return n_folds
    
    def split(self, X, y=None, groups=None):
        """Generate train/test indices ensuring balanced group holdout"""
        if groups is None or y is None:
            raise ValueError("groups and y must be provided")
        
        unique_groups = np.unique(groups)
        
        # Map each group to its class label
        group_labels = {}
        for group in unique_groups:
            group_mask = groups == group
            group_labels[group] = y[group_mask][0]
        
        # Separate subjects by class
        class_0_subjects = [g for g, label in group_labels.items() if label == 0]
        class_1_subjects = [g for g, label in group_labels.items() if label == 1]
        
        logger.info(f"Class 0 (control) subjects: {len(class_0_subjects)}")
        logger.info(f"Class 1 (exercised) subjects: {len(class_1_subjects)}")
        
        # Generate all combinations of 2 subjects from each class
        class_0_pairs = list(combinations(class_0_subjects, 2))
        class_1_pairs = list(combinations(class_1_subjects, 2))
        
        fold_count = 0
        # Create folds by combining one pair from each class
        for c0_pair in class_0_pairs:
            for c1_pair in class_1_pairs:
                test_groups = set(c0_pair + c1_pair)  # 4 subjects total: 2 + 2
                
                # Create masks
                test_mask = np.isin(groups, list(test_groups))
                train_mask = ~test_mask
                
                train_indices = np.where(train_mask)[0]
                test_indices = np.where(test_mask)[0]
                
                fold_count += 1
                
                yield train_indices, test_indices
        
        logger.info(f"Generated {fold_count} balanced CV folds")
    
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generate boolean masks for test sets"""
        for train_idx, test_idx in self.split(X, y, groups):
            test_mask = np.zeros(len(groups), dtype=bool)
            test_mask[test_idx] = True
            yield test_mask

def bootstrap_confidence_intervals(accuracies, confidence=0.95):
    """Calculate confidence intervals using t-distribution for small samples"""
    mean_acc = np.mean(accuracies)
    std_err = stats.sem(accuracies)
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(accuracies)-1)
    return mean_acc - h, mean_acc + h

def cohens_d_effect_size(group1, group2):
    """Calculate Cohen's d effect size between two groups"""
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def assess_multicollinearity(X, feature_names, threshold=5.0):
    """Identify features with high variance inflation factors"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
    return vif_data, high_vif_features

def iterative_feature_selection(X, y, groups, feature_names, vif_threshold=5.0, max_features=15):
    """Remove multicollinear features using iterative VIF analysis and importance ranking"""
    logger.info("Beginning iterative feature selection process...")
    
    # Initial feature scaling for VIF calculation
    X_scaled = StandardScaler().fit_transform(X)
    remaining_features = list(range(len(feature_names)))
    remaining_names = feature_names.copy()
    
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"Feature selection iteration {iteration}: {len(remaining_features)} features")
        
        if len(remaining_features) <= 2:
            break
            
        # Calculate VIF for current feature set
        vif_scores = []
        for i in range(len(remaining_features)):
            try:
                vif = variance_inflation_factor(X_scaled[:, remaining_features], i)
                vif_scores.append(vif)
            except:
                vif_scores.append(float('inf'))
        
        # Identify highest VIF feature
        max_vif_idx = np.argmax(vif_scores)
        max_vif = vif_scores[max_vif_idx]
        
        if max_vif <= vif_threshold:
            break
            
        # Remove feature with highest VIF
        removed_feature = remaining_names[max_vif_idx]
        logger.info(f"  Removing {removed_feature} (VIF: {max_vif:.2f})")
        
        remaining_features.pop(max_vif_idx)
        remaining_names.pop(max_vif_idx)
    
    # Rank remaining features by importance if needed
    if len(remaining_features) > max_features:
        logger.info("Ranking features by model importance...")
        X_subset = X_scaled[:, remaining_features]
        
        # Cross-validated importance estimation using BalancedLeaveTwoGroupsOut
        cv = BalancedLeaveTwoGroupsOut()
        importance_scores = []
        
        for train_idx, test_idx in cv.split(X_subset, y, groups):
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_subset[train_idx], y[train_idx])
            importance_scores.append(model.feature_importances_)
        
        # Select top features by average importance
        mean_importance = np.mean(importance_scores, axis=0)
        importance_ranking = np.argsort(mean_importance)[::-1]
        final_features = [remaining_features[i] for i in importance_ranking[:max_features]]
        final_names = [remaining_names[i] for i in importance_ranking[:max_features]]
        
        logger.info(f"Selected {len(final_features)} most important features:")
        for i, (feat_idx, name) in enumerate(zip(final_features, final_names)):
            importance = mean_importance[importance_ranking[i]]
            logger.info(f"  {i+1}. {name}: {importance:.4f}")
    else:
        final_features = remaining_features
        final_names = remaining_names
    
    return final_features, final_names

def verify_group_isolation(cv_splitter, groups, y):
    """Ensure no subject appears in both training and test sets within any fold"""
    logger.info("Verifying cross-validation group isolation...")
    logger.info("Expected: 4 test subjects per fold (2 control + 2 exercised)")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(np.zeros(len(groups)), y, groups)):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])
        
        overlap = train_subjects.intersection(test_subjects)
        if overlap:
            raise ValueError(f"Fold {fold_idx}: Subjects {overlap} appear in both training and test sets!")
        
        # Count subjects per class in test set
        test_class_0_subjects = [s for s in test_subjects if y[groups == s][0] == 0]
        test_class_1_subjects = [s for s in test_subjects if y[groups == s][0] == 1]
        
        # Count subjects per class in train set
        train_class_0_subjects = [s for s in train_subjects if y[groups == s][0] == 0]
        train_class_1_subjects = [s for s in train_subjects if y[groups == s][0] == 1]
        
        logger.info(f"\nFold {fold_idx + 1}:")
        logger.info(f"  TEST SET: {len(test_subjects)} subjects total")
        logger.info(f"    - Control subjects (class 0): {len(test_class_0_subjects)} → {test_class_0_subjects}")
        logger.info(f"    - Exercised subjects (class 1): {len(test_class_1_subjects)} → {test_class_1_subjects}")
        logger.info(f"  TRAIN SET: {len(train_subjects)} subjects total")
        logger.info(f"    - Control subjects (class 0): {len(train_class_0_subjects)}")
        logger.info(f"    - Exercised subjects (class 1): {len(train_class_1_subjects)}")
        
        # Verify exactly 2 from each class in test set
        if len(test_class_0_subjects) != 2 or len(test_class_1_subjects) != 2:
            raise ValueError(f"Fold {fold_idx}: Expected 2 subjects from each class in test set, "
                           f"got {len(test_class_0_subjects)} from class 0 and {len(test_class_1_subjects)} from class 1")
        
        # Verify total of 4 test subjects
        if len(test_subjects) != 4:
            raise ValueError(f"Fold {fold_idx}: Expected 4 test subjects, got {len(test_subjects)}")
    
    logger.info("\n✓ Group isolation verified - proper balanced cross-validation structure maintained")
    logger.info("✓ Each fold uses exactly 4 test subjects (2 control + 2 exercised)")

def identify_signal_artifacts(signal_data, fs=2500):
    """Detect potential artifacts in neural signal recordings"""
    artifacts = {}
    
    # Check for flat signal segments (equipment failure)
    flat_threshold = np.std(signal_data) * 0.01
    artifacts['flat_segments'] = np.std(signal_data) < flat_threshold
    
    # Detect extreme amplitude outliers
    z_scores = np.abs(stats.zscore(signal_data))
    artifacts['extreme_outliers'] = np.sum(z_scores > 4) / len(signal_data) > 0.05
    
    # Check for excessive high-frequency content
    freqs, psd = signal.welch(signal_data, fs=fs)
    high_freq_power = np.sum(psd[freqs > 500]) / np.sum(psd)
    artifacts['high_freq_noise'] = high_freq_power > 0.5
    
    return artifacts

def theta_gamma_phase_amplitude_coupling(signal_data, fs=2500):
    """Calculate phase-amplitude coupling between theta (5-7 Hz) and gamma (35-80 Hz) bands"""
    try:
        if len(signal_data) < 2500:  # Minimum 1 second of data required
            return np.nan
        
        # Validate signal quality
        if np.std(signal_data) < 1e-10 or np.all(signal_data == signal_data[0]):
            return np.nan
        
        nyquist = fs / 2
        
        # Define frequency bands for phase-amplitude coupling
        theta_low = 5 / nyquist
        theta_high = 7 / nyquist
        gamma_low = 35 / nyquist
        gamma_high = min(80 / nyquist, 0.9)
        
        # Validate frequency bounds
        if theta_high >= 0.95 or theta_low <= 0.01 or gamma_high <= gamma_low:
            return np.nan
        
        # Design stable bandpass filters
        filter_order = 3
        
        # Theta band filter for phase extraction
        try:
            b_theta, a_theta = signal.butter(filter_order, [theta_low, theta_high], btype='band')
            
            # Check filter stability
            if len(a_theta) > 1:
                poles = np.roots(a_theta)
                if np.any(np.abs(poles) >= 0.99):
                    # Fallback to lower order if unstable
                    b_theta, a_theta = signal.butter(2, [theta_low, theta_high], btype='band')
                    poles = np.roots(a_theta)
                    if np.any(np.abs(poles) >= 0.99):
                        return np.nan
            
            # Apply filter with edge padding
            pad_length = min(len(signal_data) // 4, 500)
            theta_signal = signal.filtfilt(b_theta, a_theta, signal_data, padlen=pad_length)
            
        except Exception:
            return np.nan
        
        # Gamma band filter for amplitude extraction
        try:
            b_gamma, a_gamma = signal.butter(filter_order, [gamma_low, gamma_high], btype='band')
            
            # Check filter stability
            if len(a_gamma) > 1:
                poles = np.roots(a_gamma)
                if np.any(np.abs(poles) >= 0.99):
                    # Fallback to lower order if unstable
                    b_gamma, a_gamma = signal.butter(2, [gamma_low, gamma_high], btype='band')
                    poles = np.roots(a_gamma)
                    if np.any(np.abs(poles) >= 0.99):
                        return np.nan
            
            # Apply filter with edge padding
            pad_length = min(len(signal_data) // 4, 500)
            gamma_signal = signal.filtfilt(b_gamma, a_gamma, signal_data, padlen=pad_length)
            
        except Exception:
            return np.nan
        
        # Validate filtered signals
        if (np.any(np.isnan(theta_signal)) or np.any(np.isnan(gamma_signal)) or
            np.std(theta_signal) < 1e-10 or np.std(gamma_signal) < 1e-10):
            return np.nan
        
        # Extract instantaneous phase and amplitude using Hilbert transform
        try:
            theta_analytic = hilbert(theta_signal)
            theta_phase = np.angle(theta_analytic)
            
            gamma_analytic = hilbert(gamma_signal)
            gamma_amplitude = np.abs(gamma_analytic)
        except Exception:
            return np.nan
        
        # Validate Hilbert transform results
        if np.any(np.isnan(theta_phase)) or np.any(np.isnan(gamma_amplitude)):
            return np.nan
        
        # Calculate Modulation Index using phase binning approach
        n_bins = 18  # Standard 20-degree bins
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        mean_amplitudes = np.zeros(n_bins)
        valid_bins = 0
        
        # Calculate mean gamma amplitude for each theta phase bin
        for i in range(n_bins):
            phase_mask = (theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1])
            bin_count = np.sum(phase_mask)
            
            if bin_count > 0:
                mean_amplitudes[i] = np.mean(gamma_amplitude[phase_mask])
                valid_bins += 1
            else:
                mean_amplitudes[i] = 0
        
        # Require sufficient data coverage across phase bins
        if valid_bins < n_bins // 2:
            return np.nan
        
        # Normalize amplitudes to create probability distribution
        total_amplitude = np.sum(mean_amplitudes)
        if total_amplitude <= 0:
            return np.nan
        
        P = mean_amplitudes / total_amplitude
        P_nonzero = P[P > 1e-10]  # Remove near-zero probabilities
        
        if len(P_nonzero) < 3:
            return np.nan
        
        # Calculate Modulation Index using Kullback-Leibler divergence
        try:
            uniform_entropy = np.log(n_bins)  # Maximum entropy for uniform distribution
            actual_entropy = -np.sum(P_nonzero * np.log(P_nonzero))  # Shannon entropy
            
            # Normalized modulation index (0 = uniform, 1 = maximum coupling)
            modulation_index = (uniform_entropy - actual_entropy) / uniform_entropy
            
            # Validate and constrain result
            if np.isnan(modulation_index) or np.isinf(modulation_index):
                return np.nan
            
            return np.clip(modulation_index, 0.0, 1.0)
            
        except Exception:
            return np.nan
        
    except Exception:
        return np.nan

def extract_neural_features(signal_data, window_size, fs=2500):
    """Extract comprehensive feature set from neural signal window"""
    signal_data = np.asarray(signal_data, dtype=np.float64)
    signal_data = signal_data[np.isfinite(signal_data)]
    
    if len(signal_data) < window_size:
        return None
    
    # Check for signal quality issues
    artifacts = identify_signal_artifacts(signal_data, fs)
    if any(artifacts.values()):
        logger.debug(f"Signal artifacts detected: {artifacts}")
    
    features = {}
    
    # Time-domain statistical features
    features['min'] = np.min(signal_data)
    features['max'] = np.max(signal_data)
    features['zero_crossings'] = np.sum(np.diff(np.signbit(signal_data)))
    features['energy'] = np.sum(signal_data**2)
    features['rms'] = np.sqrt(np.mean(signal_data**2))
    
    # Signal variability measures
    diff_signal = np.diff(signal_data)
    features['signal_variability'] = np.std(diff_signal)
    features['diff_variance'] = np.var(diff_signal)
    
    # Peak detection and morphology
    try:
        threshold = np.mean(signal_data) + 0.5 * np.std(signal_data)
        peaks, _ = find_peaks(signal_data, height=threshold)
        valleys, _ = find_peaks(-signal_data, height=-np.mean(signal_data))
        features['num_peaks'] = len(peaks)
        features['peak_valley_ratio'] = len(peaks) / (len(valleys) + 1)
        
        if len(peaks) > 0:
            peak_heights = signal_data[peaks]
            features['std_peak_height'] = np.std(peak_heights)
        else:
            features['std_peak_height'] = 0
    except:
        features['num_peaks'] = 0
        features['peak_valley_ratio'] = 0
        features['std_peak_height'] = 0
    
    # Cross-frequency coupling analysis
    features['phase_amplitude_coupling'] = theta_gamma_phase_amplitude_coupling(signal_data, fs)
    
    # Spectral power analysis
    try:
        nperseg = min(512, len(signal_data)//4)
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
        
        nyquist = fs / 2
        frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'slow_gamma': (30, 60),
            'fast_gamma': (60, 100),
            'high_freq': (100, min(300, nyquist-1))
        }
        
        def calculate_band_power(freqs, psd, band):
            if band[1] >= nyquist:
                return 0
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0
        
        # Calculate absolute power in each frequency band
        for band_name, band_range in frequency_bands.items():
            features[f'{band_name}_power'] = calculate_band_power(freqs, psd, band_range)
        
        # Calculate relative power features
        total_power = np.trapz(psd, freqs)
        if total_power > 0:
            features['high_freq_rel'] = features['high_freq_power'] / total_power
            features['slow_gamma_rel'] = features['slow_gamma_power'] / total_power
        else:
            features['high_freq_rel'] = 0
            features['slow_gamma_rel'] = 0
        
        # Calculate cross-band power ratios
        features['fast_gamma_slow_gamma_ratio'] = features['fast_gamma_power'] / (features['slow_gamma_power'] + 1e-10)
        features['gamma_beta_ratio'] = (features['slow_gamma_power'] + features['fast_gamma_power']) / (features['beta_power'] + 1e-10)
        features['fast_slow_gamma_ratio'] = features['fast_gamma_power'] / (features['slow_gamma_power'] + 1e-10)
        features['slow_gamma_theta_ratio'] = features['slow_gamma_power'] / (features['theta_power'] + 1e-10)
        features['fast_gamma_theta_ratio'] = features['fast_gamma_power'] / (features['theta_power'] + 1e-10)
        
    except Exception as e:
        logger.warning(f"Spectral analysis failed: {e}")
        # Set default values for failed spectral features
        spectral_features = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 
                           'slow_gamma_power', 'fast_gamma_power', 'high_freq_power',
                           'high_freq_rel', 'slow_gamma_rel', 'fast_gamma_slow_gamma_ratio',
                           'gamma_beta_ratio', 'fast_slow_gamma_ratio', 'slow_gamma_theta_ratio',
                           'fast_gamma_theta_ratio']
        for feat in spectral_features:
            features[feat] = 0
    
    # Clean up any remaining invalid values
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0
    
    return features

def load_experimental_data(file_path):
    """Load neural signal data from MATLAB file format"""
    logger.info("Loading experimental neural data...")
    data = scipy.io.loadmat(file_path)
    g_s = data['G']['s'][0, 0]
    
    control_subjects = []
    exercised_subjects = []
    
    # Extract control group recordings
    for i in range(g_s.shape[1]):
        raw_data = g_s[0, i].flatten()
        raw_data = np.asarray(raw_data, dtype=np.float64)
        raw_data = raw_data[np.isfinite(raw_data)]  # Remove invalid values
        if len(raw_data) > 0:
            control_subjects.append(raw_data)
    
    # Extract exercised group recordings
    for i in range(g_s.shape[1]):
        raw_data = g_s[1, i].flatten()
        raw_data = np.asarray(raw_data, dtype=np.float64)
        raw_data = raw_data[np.isfinite(raw_data)]  # Remove invalid values
        if len(raw_data) > 0:
            exercised_subjects.append(raw_data)
    
    logger.info(f"Loaded data from {len(control_subjects)} control and {len(exercised_subjects)} exercised subjects")
    return control_subjects, exercised_subjects

def build_windowed_dataset(control_subjects, exercised_subjects, window_size, overlap):
    """Create feature dataset using sliding window approach"""
    logger.info(f"Building dataset with {window_size}-point windows at {overlap:.1%} overlap")
    
    step_size = int(window_size * (1 - overlap))
    
    feature_matrix = []
    class_labels = []
    subject_groups = []
    
    # Process control subjects
    logger.info("Processing control subject recordings...")
    for i, subject_data in enumerate(control_subjects):
        subject_id = f'control_{i}'
        subject_windows = []
        
        # Extract overlapping windows from subject's recording
        for start in range(0, len(subject_data) - window_size + 1, step_size):
            window = subject_data[start:start + window_size]
            if len(window) == window_size:
                features = extract_neural_features(window, window_size)
                if features is not None:
                    subject_windows.append(list(features.values()))
        
        logger.info(f"  Subject {i+1}/{len(control_subjects)} - extracted {len(subject_windows)} windows")
        
        # Add windows to dataset
        for window_features in subject_windows:
            feature_matrix.append(window_features)
            class_labels.append(0)  # Control class
            subject_groups.append(subject_id)
    
    # Process exercised subjects
    logger.info("Processing exercised subject recordings...")
    for i, subject_data in enumerate(exercised_subjects):
        subject_id = f'exercised_{i}'
        subject_windows = []
        
        # Extract overlapping windows from subject's recording
        for start in range(0, len(subject_data) - window_size + 1, step_size):
            window = subject_data[start:start + window_size]
            if len(window) == window_size:
                features = extract_neural_features(window, window_size)
                if features is not None:
                    subject_windows.append(list(features.values()))
        
        logger.info(f"  Subject {i+1}/{len(exercised_subjects)} - extracted {len(subject_windows)} windows")
        
        # Add windows to dataset
        for window_features in subject_windows:
            feature_matrix.append(window_features)
            class_labels.append(1)  # Exercised class
            subject_groups.append(subject_id)
    
    # Define feature names for interpretability
    sample_window = control_subjects[0][:window_size] if len(control_subjects[0]) >= window_size else control_subjects[0]
    sample_features = extract_neural_features(sample_window, window_size)
    feature_names = [
        'min', 'max', 'zero_crossings', 'energy', 'rms', 'signal_variability', 
        'diff_variance', 'num_peaks', 'peak_valley_ratio', 'std_peak_height',
        'phase_amplitude_coupling', 'delta_power', 'theta_power', 'alpha_power',
        'beta_power', 'slow_gamma_power', 'fast_gamma_power', 'high_freq_power',
        'high_freq_rel', 'slow_gamma_rel', 'fast_gamma_slow_gamma_ratio',
        'gamma_beta_ratio', 'fast_slow_gamma_ratio', 'slow_gamma_theta_ratio',
        'fast_gamma_theta_ratio'
    ]
    
    return np.array(feature_matrix), np.array(class_labels), np.array(subject_groups), feature_names

def cross_validate_model(X, y, groups, model_params):
    """Evaluate model performance using Balanced Leave-Two-Groups-Out cross-validation"""
    cv = BalancedLeaveTwoGroupsOut()
    
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        # Partition data for current fold
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Apply feature scaling within fold to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # Train classifier
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
    
    return np.mean(fold_accuracies)

def parallel_permutation_test(X, y, groups, model_params, n_permutations=1000):
    """Statistical significance testing using parallel permutation approach"""
    from joblib import Parallel, delayed
    import multiprocessing
    
    logger.info(f"Running permutation test with {n_permutations} iterations...")
    logger.info(f"Utilizing {multiprocessing.cpu_count()} CPU cores for parallel processing")
    
    # Establish baseline performance
    baseline_accuracy = cross_validate_model(X, y, groups, model_params)
    
    # Map subjects to their true class labels
    unique_subjects = np.unique(groups)
    subject_labels = {}
    for subject in unique_subjects:
        subject_mask = groups == subject
        subject_labels[subject] = y[subject_mask][0]
    
    def evaluate_single_permutation(permutation_idx, random_seed):
        """Evaluate model performance on one permuted dataset"""
        np.random.seed(random_seed + permutation_idx)  # Ensure reproducible randomness
        
        # Randomly reassign class labels to subjects
        subjects = list(subject_labels.keys())
        original_labels = list(subject_labels.values())
        shuffled_labels = np.random.permutation(original_labels)
        
        # Create new subject-to-label mapping
        permuted_subject_labels = dict(zip(subjects, shuffled_labels))
        
        # Apply permuted labels to all windows
        y_permuted = np.array([permuted_subject_labels[subject] for subject in groups])
        
        # Use streamlined model parameters for computational efficiency
        efficient_params = {
            'n_estimators': min(50, model_params.get('n_estimators', 50)),
            'max_depth': min(8, model_params.get('max_depth', 8)),
            'learning_rate': max(0.1, model_params.get('learning_rate', 0.1)),
            'subsample': model_params.get('subsample', 0.8),
            'random_state': 42
        }
        
        # Evaluate model with permuted labels
        return evaluate_model_efficiently(X, y_permuted, groups, efficient_params)
    
    # Execute permutations in parallel
    logger.info("Beginning parallel permutation evaluation...")
    permutation_accuracies = Parallel(n_jobs=-1, verbose=10, backend='threading')(
        delayed(evaluate_single_permutation)(i, 42) for i in range(n_permutations)
    )
    
    # Calculate statistical significance
    p_value = (np.sum(np.array(permutation_accuracies) >= baseline_accuracy) + 1) / (n_permutations + 1)
    
    logger.info(f"Permutation test results:")
    logger.info(f"  Baseline accuracy: {baseline_accuracy:.4f}")
    logger.info(f"  Null distribution: {np.mean(permutation_accuracies):.4f} ± {np.std(permutation_accuracies):.4f}")
    logger.info(f"  P-value: {p_value:.4f}")
    logger.info(f"  Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    return baseline_accuracy, permutation_accuracies, p_value

def evaluate_model_efficiently(X, y, groups, model_params):
    """Streamlined model evaluation for permutation testing"""
    cv = BalancedLeaveTwoGroupsOut()
    
    fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        # Partition data
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Scale features within fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # Train and evaluate model
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
    
    return np.mean(fold_accuracies)

def comprehensive_model_assessment(X, y, groups, feature_names, model_params, apply_feature_selection=True):
    """Comprehensive evaluation of model performance and reliability"""
    
    logger.info("="*70)
    logger.info("COMPREHENSIVE MODEL ASSESSMENT")
    logger.info("="*70)
    
    # Apply feature selection if requested
    if apply_feature_selection:
        logger.info("Applying feature selection to reduce multicollinearity...")
        selected_indices, selected_names = iterative_feature_selection(
            X, y, groups, feature_names, vif_threshold=5.0, max_features=15
        )
        X_selected = X[:, selected_indices]
        feature_names_selected = selected_names
        logger.info(f"Using {len(selected_names)} selected features for analysis")
    else:
        X_selected = X
        feature_names_selected = feature_names
        logger.info("Using complete feature set without selection")
    
    # Set up cross-validation framework using Balanced Leave-Two-Groups-Out
    cv = BalancedLeaveTwoGroupsOut()
    verify_group_isolation(cv, groups, y)
    
    # Initialize performance tracking
    performance_metrics = {
        'accuracies': [],
        'sensitivities': [],
        'specificities': [],
        'aucs': [],
        'feature_importances': []
    }
    
    # Perform cross-validation evaluation
    logger.info("Conducting Balanced Leave-Two-Groups-Out cross-validation assessment...")
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_selected, y, groups)):
        logger.info(f"Evaluating fold {fold + 1}...")
        
        # Partition data for current fold
        X_train_raw = X_selected[train_idx]
        X_test_raw = X_selected[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Check if test set has both classes
        unique_test_classes = np.unique(y_test)
        unique_train_classes = np.unique(y_train)
        
        if len(unique_test_classes) < 2:
            logger.warning(f"Fold {fold + 1}: Test set contains only class {unique_test_classes[0]} - skipping fold")
            continue
            
        if len(unique_train_classes) < 2:
            logger.warning(f"Fold {fold + 1}: Training set contains only class {unique_train_classes[0]} - skipping fold")
            continue
        
        # Apply feature scaling within fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # Train classifier
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Robust confusion matrix handling
        cm = confusion_matrix(y_test, y_pred)
        
        # Handle different confusion matrix sizes
        if cm.size == 4:  # Standard 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:  # Single class case
            if unique_test_classes[0] == 0:  # Only class 0 in test
                if y_pred[0] == 0:  # Predicted correctly
                    tn, fp, fn, tp = len(y_test), 0, 0, 0
                else:  # Predicted incorrectly
                    tn, fp, fn, tp = 0, len(y_test), 0, 0
            else:  # Only class 1 in test
                if y_pred[0] == 1:  # Predicted correctly
                    tn, fp, fn, tp = 0, 0, 0, len(y_test)
                else:  # Predicted incorrectly
                    tn, fp, fn, tp = 0, 0, len(y_test), 0
        else:
            logger.warning(f"Fold {fold + 1}: Unexpected confusion matrix size {cm.size} - using default values")
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Calculate sensitivity and specificity with safe division
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Area under ROC curve with error handling
        try:
            if len(unique_test_classes) >= 2:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = np.nan
        except Exception as e:
            logger.warning(f"Fold {fold + 1}: AUC calculation failed: {e}")
            auc = np.nan
        
        # Store fold results
        performance_metrics['accuracies'].append(accuracy)
        performance_metrics['sensitivities'].append(sensitivity)
        performance_metrics['specificities'].append(specificity)
        performance_metrics['aucs'].append(auc)
        performance_metrics['feature_importances'].append(model.feature_importances_)
        
        logger.info(f"  Fold {fold + 1}: Acc={accuracy:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}, AUC={auc:.4f}")
    
    # Filter out NaN values from AUC calculations for summary statistics
    valid_aucs = [auc for auc in performance_metrics['aucs'] if not np.isnan(auc)]
    if len(valid_aucs) < len(performance_metrics['aucs']):
        logger.warning(f"Excluded {len(performance_metrics['aucs']) - len(valid_aucs)} NaN AUC values from summary")
        performance_metrics['aucs'] = valid_aucs
    
    # Ensure we have enough valid folds
    if len(performance_metrics['accuracies']) == 0:
        logger.error("No valid folds completed - cannot generate performance summary")
        return None, None
    
    # Summarize performance across folds
    results_summary = {}
    for metric_name, values in performance_metrics.items():
        if metric_name != 'feature_importances':
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_lower, ci_upper = bootstrap_confidence_intervals(values)
            
            results_summary[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': values
            }
    
    # Analyze feature importance
    if len(performance_metrics['feature_importances']) > 0:
        mean_feature_importance = np.mean(performance_metrics['feature_importances'], axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names_selected,
            'importance': mean_feature_importance
        }).sort_values('importance', ascending=False)
    else:
        importance_df = pd.DataFrame({'feature': feature_names_selected, 'importance': [0]*len(feature_names_selected)})
    
    # Log summary results
    logger.info("\n" + "="*50)
    logger.info("BALANCED LEAVE-TWO-GROUPS-OUT CROSS-VALIDATION PERFORMANCE SUMMARY")
    logger.info("="*50)
    
    for metric_name, stats in results_summary.items():
        logger.info(f"{metric_name.upper()}:")
        logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
    
    logger.info(f"\nMOST IMPORTANT FEATURES:")
    for i, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return results_summary, importance_df

def analyze_effect_sizes(X, y, groups, feature_names):
    """Calculate effect sizes between experimental groups"""
    logger.info("\nAnalyzing effect sizes between groups...")
    
    # Calculate per-subject feature averages
    control_subject_features = []
    exercised_subject_features = []
    
    unique_subjects = np.unique(groups)
    for subject in unique_subjects:
        subject_mask = groups == subject
        subject_label = y[subject_mask][0]
        subject_features = np.mean(X[subject_mask], axis=0)
        
        if subject_label == 0:
            control_subject_features.append(subject_features)
        else:
            exercised_subject_features.append(subject_features)
    
    control_features = np.array(control_subject_features)
    exercised_features = np.array(exercised_subject_features)
    
    # Calculate Cohen's d for each feature
    effect_size_results = []
    for i, feature_name in enumerate(feature_names):
        cohens_d = cohens_d_effect_size(exercised_features[:, i], control_features[:, i])
        effect_size_results.append({
            'feature': feature_name,
            'cohens_d': cohens_d,
            'magnitude': 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small'
        })
    
    effect_sizes_df = pd.DataFrame(effect_size_results).sort_values('cohens_d', key=abs, ascending=False)
    
    # Calculate overall effect size
    overall_control = np.mean(control_features, axis=1)
    overall_exercised = np.mean(exercised_features, axis=1)
    overall_cohens_d = cohens_d_effect_size(overall_exercised, overall_control)
    
    logger.info(f"Overall effect size (Cohen's d): {overall_cohens_d:.4f}")
    magnitude_interpretation = "Large" if abs(overall_cohens_d) >= 0.8 else "Medium" if abs(overall_cohens_d) >= 0.5 else "Small"
    logger.info(f"Effect magnitude: {magnitude_interpretation}")
    
    logger.info(f"\nLargest effect sizes by feature:")
    for i, row in effect_sizes_df.head().iterrows():
        logger.info(f"  {row['feature']}: {row['cohens_d']:.4f} ({row['magnitude']})")
    
    return effect_sizes_df, overall_cohens_d

def generate_evaluation_plots(results_summary, importance_df, effect_sizes_df):
    """Create comprehensive visualization of model evaluation results"""
    logger.info("Generating evaluation visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Performance metrics bar chart
    metrics = ['accuracies', 'sensitivities', 'specificities', 'aucs']
    means = [results_summary[m]['mean'] for m in metrics]
    stds = [results_summary[m]['std'] for m in metrics]
    
    axes[0, 0].bar(metrics, means, yerr=stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title('Balanced Leave-Two-Groups-Out Performance Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    for i, (mean, std) in enumerate(zip(means, stds)):
        axes[0, 0].text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                       ha='center', va='bottom')
    
    # Feature importance ranking
    top_features = importance_df.head(10)
    axes[0, 1].barh(range(len(top_features)), top_features['importance'])
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_title('Top 10 Feature Importance Rankings')
    axes[0, 1].set_xlabel('Importance Score')
    
    # Effect size visualization
    top_effects = effect_sizes_df.head(10)
    colors = ['red' if d < 0 else 'blue' for d in top_effects['cohens_d']]
    axes[0, 2].barh(range(len(top_effects)), top_effects['cohens_d'], color=colors, alpha=0.7)
    axes[0, 2].set_yticks(range(len(top_effects)))
    axes[0, 2].set_yticklabels(top_effects['feature'])
    axes[0, 2].set_title("Top 10 Effect Sizes (Cohen's d)")
    axes[0, 2].set_xlabel("Cohen's d")
    axes[0, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Accuracy distribution histogram
    axes[1, 0].hist(results_summary['accuracies']['values'], bins=8, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(results_summary['accuracies']['mean'], color='red', linestyle='--', 
                      label=f"Mean: {results_summary['accuracies']['mean']:.3f}")
    axes[1, 0].set_title('Accuracy Distribution Across Folds')
    axes[1, 0].set_xlabel('Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Performance across folds
    fold_indices = range(1, len(results_summary['accuracies']['values']) + 1)
    axes[1, 1].plot(fold_indices, results_summary['accuracies']['values'], 'o-', label='Accuracy')
    axes[1, 1].plot(fold_indices, results_summary['sensitivities']['values'], 's-', label='Sensitivity')
    axes[1, 1].plot(fold_indices, results_summary['specificities']['values'], '^-', label='Specificity')
    axes[1, 1].plot(fold_indices, results_summary['aucs']['values'], 'd-', label='AUC')
    axes[1, 1].set_title('Performance Metrics Across CV Folds')
    axes[1, 1].set_xlabel('Fold Number')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Confidence interval visualization
    metrics_ci = ['accuracies', 'sensitivities', 'specificities', 'aucs']
    means_ci = [results_summary[m]['mean'] for m in metrics_ci]
    ci_lower = [results_summary[m]['ci_lower'] for m in metrics_ci]
    ci_upper = [results_summary[m]['ci_upper'] for m in metrics_ci]
    
    x_pos = range(len(metrics_ci))
    axes[1, 2].errorbar(x_pos, means_ci, 
                       yerr=[np.array(means_ci) - np.array(ci_lower), 
                             np.array(ci_upper) - np.array(means_ci)],
                       fmt='o', capsize=5, capthick=2)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(metrics_ci)
    axes[1, 2].set_title('95% Confidence Intervals')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_classifier_evaluation_lm.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main evaluation pipeline for neural signal classification"""
    logger.info("="*70)
    logger.info("NEURAL SIGNAL CLASSIFICATION EVALUATION")
    logger.info("="*70)
    
    # Load experimental configuration
    config_file = 'optuna_lm_best_config.json'
    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} not found!")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    logger.info("Loaded experimental configuration:")
    logger.info(f"  Window size: {config['best_window_size']} samples")
    logger.info(f"  Window overlap: {config['best_overlap']:.1%}")
    logger.info(f"  Model parameters: {len(config['best_model_params'])} hyperparameters")
    
    # Load experimental data
    data_file = 'D:/Laboratorio/Registros/Experimental-ejercicio/LM8-todo-nuevo.mat'
    control_subjects, exercised_subjects = load_experimental_data(data_file)
    
    # Build feature dataset
    X, y, groups, feature_names = build_windowed_dataset(
        control_subjects, exercised_subjects, 
        config['best_window_size'], 
        config['best_overlap']
    )
    
    logger.info(f"Dataset characteristics:")
    logger.info(f"  Total samples: {len(X)}")
    logger.info(f"  Feature dimensions: {len(feature_names)}")
    logger.info(f"  Control samples: {sum(y==0)}")
    logger.info(f"  Exercised samples: {sum(y==1)}")
    logger.info(f"  Unique subjects: {len(np.unique(groups))}")
    
    # Calculate number of Balanced Leave-Two-Groups-Out folds
    unique_groups = np.unique(groups)
    group_labels = {}
    for group in unique_groups:
        group_mask = groups == group
        group_labels[group] = y[group_mask][0]
    
    class_0_subjects = [g for g, label in group_labels.items() if label == 0]
    class_1_subjects = [g for g, label in group_labels.items() if label == 1]
    
    from math import comb
    n_folds = comb(len(class_0_subjects), 2) * comb(len(class_1_subjects), 2)
    logger.info(f"  Balanced Leave-Two-Groups-Out will create {n_folds} folds")
    logger.info(f"  Each fold: 4 test subjects (2 control + 2 exercised)")
    
    # Comprehensive model evaluation
    results_summary, importance_df = comprehensive_model_assessment(
        X, y, groups, feature_names, config['best_model_params']
    )
    
    # Statistical significance testing
    baseline_score, permutation_scores, p_value = parallel_permutation_test(
        X, y, groups, config['best_model_params'], n_permutations=1000
    )
    
    # Effect size analysis
    effect_sizes_df, overall_cohens_d = analyze_effect_sizes(X, y, groups, feature_names)
    
    # Generate visualizations
    generate_evaluation_plots(results_summary, importance_df, effect_sizes_df)
    
    # Compile comprehensive results
    evaluation_results = {
        'cross_validation_performance': results_summary,
        'permutation_test_results': {
            'baseline_score': baseline_score,
            'p_value': p_value,
            'null_distribution_mean': float(np.mean(permutation_scores)),
            'null_distribution_std': float(np.std(permutation_scores))
        },
        'effect_size_analysis': {
            'overall_cohens_d': overall_cohens_d,
            'feature_level_effects': effect_sizes_df.to_dict('records')
        },
        'feature_importance_rankings': importance_df.to_dict('records'),
        'dataset_characteristics': {
            'total_samples': len(X),
            'feature_count': len(feature_names),
            'control_samples': int(sum(y==0)),
            'exercised_samples': int(sum(y==1)),
            'subject_count': len(np.unique(groups)),
            'cv_method': 'Balanced Leave-Two-Groups-Out',
            'n_folds': n_folds,
            'test_subjects_per_fold': 4,
            'control_test_per_fold': 2,
            'exercised_test_per_fold': 2
        }
    }
    
    # Save evaluation results
    with open('neural_classifier_evaluation_lm.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    importance_df.to_csv('feature_importance_rankings_lm.csv', index=False)
    effect_sizes_df.to_csv('effect_size_analysis_lm.csv', index=False)
    
    # Final evaluation summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*70)
    
    logger.info(f"Cross-validation method: Balanced Leave-Two-Groups-Out ({n_folds} folds)")
    logger.info(f"Test subjects per fold: 4 (2 control + 2 exercised)")
    logger.info(f"Cross-validation accuracy: {results_summary['accuracies']['mean']:.4f} ± {results_summary['accuracies']['std']:.4f}")
    logger.info(f"95% Confidence interval: [{results_summary['accuracies']['ci_lower']:.4f}, {results_summary['accuracies']['ci_upper']:.4f}]")
    logger.info(f"Sensitivity: {results_summary['sensitivities']['mean']:.4f} ± {results_summary['sensitivities']['std']:.4f}")
    logger.info(f"Specificity: {results_summary['specificities']['mean']:.4f} ± {results_summary['specificities']['std']:.4f}")
    logger.info(f"Area under ROC curve: {results_summary['aucs']['mean']:.4f} ± {results_summary['aucs']['std']:.4f}")
    logger.info(f"Permutation test p-value: {p_value:.4f}")
    logger.info(f"Statistical significance: {'Yes' if p_value < 0.05 else 'No'}")
    logger.info(f"Overall effect size (Cohen's d): {overall_cohens_d:.4f}")
    
    magnitude_interpretation = "Large" if abs(overall_cohens_d) >= 0.8 else "Medium" if abs(overall_cohens_d) >= 0.5 else "Small"
    logger.info(f"Effect magnitude: {magnitude_interpretation}")
    
    logger.info(f"\nOutput files generated:")
    logger.info(f"- neural_classifier_evaluation_lm.json")
    logger.info(f"- feature_importance_rankings_lm.csv")
    logger.info(f"- effect_size_analysis_lm.csv")
    logger.info(f"- neural_classifier_evaluation_lm.png")
    
    # Scientific validity assessment
    logger.info(f"\n" + "="*50)
    logger.info("SCIENTIFIC VALIDITY ASSESSMENT")
    logger.info("="*50)
    
    # Statistical significance criterion
    if p_value < 0.05:
        logger.info("✓ Statistical significance: ACHIEVED (p < 0.05)")
    else:
        logger.info("✗ Statistical significance: NOT ACHIEVED (p >= 0.05)")
    
    # Effect size criterion
    if abs(overall_cohens_d) >= 0.5:
        logger.info("✓ Effect size: SUBSTANTIAL (Cohen's d >= 0.5)")
    else:
        logger.info("✗ Effect size: INSUFFICIENT (Cohen's d < 0.5)")
    
    # Classification performance criterion
    if results_summary['accuracies']['mean'] >= 0.7:
        logger.info("✓ Classification performance: ACCEPTABLE (>= 70%)")
    else:
        logger.info("✗ Classification performance: SUBOPTIMAL (< 70%)")
    
    # Estimation precision criterion
    ci_width = results_summary['accuracies']['ci_upper'] - results_summary['accuracies']['ci_lower']
    if ci_width <= 0.2:
        logger.info("✓ Estimation precision: ADEQUATE (CI width <= 0.2)")
    else:
        logger.info("✗ Estimation precision: LIMITED (CI width > 0.2)")
    
    # Performance balance criterion
    sens_spec_diff = abs(results_summary['sensitivities']['mean'] - results_summary['specificities']['mean'])
    if sens_spec_diff <= 0.2:
        logger.info("✓ Performance balance: MAINTAINED (|sensitivity-specificity| <= 0.2)")
    else:
        logger.info("✗ Performance balance: IMBALANCED (|sensitivity-specificity| > 0.2)")
    
    # Discriminative capacity criterion
    if results_summary['aucs']['mean'] >= 0.75:
        logger.info("✓ Discriminative capacity: SUFFICIENT (AUC >= 0.75)")
    else:
        logger.info("✗ Discriminative capacity: INSUFFICIENT (AUC < 0.75)")
    
    # Cross-validation integrity
    logger.info("✓ Cross-validation integrity: MAINTAINED (proper group isolation)")
    logger.info("✓ Balanced test sets: 2 control + 2 exercised per fold")

if __name__ == "__main__":
    main()