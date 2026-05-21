import numpy as np
import scipy.io
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
from scipy.signal import find_peaks, hilbert
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import optuna
import logging
import os
import pickle
import hashlib
from functools import lru_cache
import json
from collections import Counter, defaultdict
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits
from sklearn.utils.validation import check_random_state
import psutil
from datetime import datetime
from itertools import combinations

warnings.filterwarnings('ignore')

# Set up comprehensive logging
def setup_comprehensive_logging():
    """Setup detailed logging for reproducibility"""
    log_filename = f"neural_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_comprehensive_logging()

# Set up logging for Optuna
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())

def monitor_memory_usage():
    """Monitor memory usage during processing"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def calculate_confidence_intervals(accuracies, confidence=0.95):
    """Calculate confidence intervals for CV accuracies"""
    mean_acc = np.mean(accuracies)
    std_err = stats.sem(accuracies)
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(accuracies)-1)
    return mean_acc - h, mean_acc + h

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                         (len(group2) - 1) * np.var(group2, ddof=1)) / 
                        (len(group1) + len(group2) - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_feature_importance(model, feature_names, X_train):
    """Analyze and visualize feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    return importance_df

def detect_multicollinearity(X, feature_names, threshold=5.0):
    """Detect multicollinearity using VIF"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    high_vif_features = vif_data[vif_data["VIF"] > threshold]["Feature"].tolist()
    return vif_data, high_vif_features

def detect_signal_artifacts(signal_data, fs=2500):
    """Detect common signal artifacts"""
    artifacts = {}
    
    # Detect flat segments (equipment failure)
    flat_threshold = np.std(signal_data) * 0.01
    artifacts['flat_segments'] = np.std(signal_data) < flat_threshold
    
    # Detect extreme outliers
    z_scores = np.abs(stats.zscore(signal_data))
    artifacts['extreme_outliers'] = np.sum(z_scores > 4) / len(signal_data) > 0.05
    
    # Detect high-frequency noise
    freqs, psd = signal.welch(signal_data, fs=fs)
    high_freq_power = np.sum(psd[freqs > 500]) / np.sum(psd)
    artifacts['high_freq_noise'] = high_freq_power > 0.3
    
    return artifacts

def permutation_test(X, y, groups, feature_names, model_params, n_permutations=1000):
    """Perform permutation test for statistical significance"""
    logger.info(f"Starting permutation test with {n_permutations} permutations")
    
    # Get baseline accuracy
    baseline_accuracy = analyze_pl_features_with_params(X, y, groups, feature_names, model_params)
    
    # Perform permutations
    perm_accuracies = []
    for i in range(n_permutations):
        if i % 100 == 0:
            logger.info(f"Permutation {i}/{n_permutations}")
        y_perm = np.random.permutation(y)
        perm_acc = analyze_pl_features_with_params(X, y_perm, groups, feature_names, model_params)
        perm_accuracies.append(perm_acc)
    
    # Calculate p-value
    p_value = np.sum(np.array(perm_accuracies) >= baseline_accuracy) / n_permutations
    logger.info(f"Permutation test completed. P-value: {p_value:.4f}")
    return p_value, perm_accuracies

class LeaveTwoAnimalsOut:
    """Leave-Two-Animals-Out cross-validation that tests on one control and one exercised animal per fold"""
    
    def __init__(self, shuffle=False, random_state=None):
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y, groups):
        """Generate train/test splits for each possible pair of animals (one from each group)"""
        # Get unique animals and their corresponding labels
        unique_groups = np.unique(groups)
        group_labels = {}
        
        for group in unique_groups:
            group_mask = groups == group
            group_labels[group] = y[group_mask][0]  # All windows from same animal have same label
        
        # Separate control and exercised animals
        control_animals = [group for group, label in group_labels.items() if label == 0]
        exercised_animals = [group for group, label in group_labels.items() if label == 1]
        
        logger.info(f"Found {len(control_animals)} control animals and {len(exercised_animals)} exercised animals")
        
        # Generate all possible pairs (one control, one exercised)
        animal_pairs = list(combinations(control_animals, 1)) + list(combinations(exercised_animals, 1))
        all_pairs = []
        
        for control_animal in control_animals:
            for exercised_animal in exercised_animals:
                all_pairs.append((control_animal, exercised_animal))
        
        if self.shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(all_pairs)
        
        logger.info(f"Generated {len(all_pairs)} animal pairs for cross-validation")
        
        # Generate train/test indices for each pair
        for pair_idx, (control_animal, exercised_animal) in enumerate(all_pairs):
            test_animals = {control_animal, exercised_animal}
            
            # Get test indices (windows from the two test animals)
            test_indices = []
            train_indices = []
            
            for i, group in enumerate(groups):
                if group in test_animals:
                    test_indices.append(i)
                else:
                    train_indices.append(i)
            
            logger.info(f"Fold {pair_idx + 1}/{len(all_pairs)}: Testing on {control_animal} (control) and {exercised_animal} (exercised)")
            logger.info(f"  Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations"""
        if groups is None:
            raise ValueError("groups parameter is required")
        
        unique_groups = np.unique(groups)
        group_labels = {}
        
        for group in unique_groups:
            # Find the first occurrence of this group to get its label
            group_indices = np.where(groups == group)[0]
            if len(group_indices) > 0:
                group_labels[group] = y[group_indices[0]]
        
        control_animals = [group for group, label in group_labels.items() if label == 0]
        exercised_animals = [group for group, label in group_labels.items() if label == 1]
        
        return len(control_animals) * len(exercised_animals)

class FeatureCacheManager:
    """Disk-based caching for feature extraction"""
    
    def __init__(self, cache_dir="feature_cache_optuna"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_stats = {'hits': 0, 'misses': 0, 'saves': 0}
    
    def _get_cache_key(self, animal_data, window_size, overlap):
        """Generate unique cache key for animal data and window configuration"""
        data_hash = hashlib.md5(str({
            'length': len(animal_data),
            'mean': float(np.mean(animal_data)),
            'std': float(np.std(animal_data)),
            'min': float(np.min(animal_data)),
            'max': float(np.max(animal_data)),
            'window_size': window_size,
            'overlap': overlap
        }).encode()).hexdigest()
        return f"features_{data_hash}.pkl"
    
    def load_features(self, animal_data, window_size, overlap):
        """Load cached features if available"""
        cache_key = self._get_cache_key(animal_data, window_size, overlap)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                self.cache_stats['hits'] += 1
                return cached_data
            except:
                pass
        
        self.cache_stats['misses'] += 1
        return None
    
    def save_features(self, animal_data, window_size, overlap, features_data):
        """Save features to disk cache"""
        cache_key = self._get_cache_key(animal_data, window_size, overlap)
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features_data, f)
            self.cache_stats['saves'] += 1
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total if total > 0 else 0
        return {**self.cache_stats, 'hit_rate': hit_rate}

# Global cache manager
cache_manager = FeatureCacheManager()

def load_pl_data(file_path):
    """Load and process pl experimental data from .mat file"""
    logger.info("Loading pl data...")
    data = scipy.io.loadmat(file_path)
    g_s = data['G']['s'][0, 0]
    
    control = []
    exercised = []
    
    # Extract control group data
    for i in range(g_s.shape[1]):
        raw_data = g_s[0, i].flatten()
        raw_data = np.asarray(raw_data, dtype=np.float64)
        raw_data = raw_data[np.isfinite(raw_data)]  # Remove NaN/inf values
        if len(raw_data) > 0:
            control.append(raw_data)
    
    # Extract exercised group data
    for i in range(g_s.shape[1]):
        raw_data = g_s[1, i].flatten()
        raw_data = np.asarray(raw_data, dtype=np.float64)
        raw_data = raw_data[np.isfinite(raw_data)]  # Remove NaN/inf values
        if len(raw_data) > 0:
            exercised.append(raw_data)
    
    logger.info(f"pl: {len(control)} control, {len(exercised)} exercised animals")
    return control, exercised

def validate_filter_stability(b, a, fs):
    """Validate filter stability and characteristics"""
    # Check filter stability
    if len(a) > 1:  # IIR filter
        poles = np.roots(a)
        if np.any(np.abs(poles) >= 1.0):
            raise ValueError("Filter is unstable - poles outside unit circle")
    
    # Check frequency response
    w, h = signal.freqz(b, a, fs=fs)
    
    # Validate filter doesn't have extreme gain
    gain_db = 20 * np.log10(np.abs(h) + 1e-12)
    if np.max(gain_db) > 20:  # More than 20dB gain
        warnings.warn("Filter has high gain - may amplify noise")
    
    return True

def validate_frequency_bands(fs, bands):
    """Add more robust frequency validation"""
    nyquist = fs / 2
    for name, (low, high) in bands.items():
        if high >= nyquist * 0.95:  # Leave margin
            warnings.warn(f"{name} band too close to Nyquist frequency")
        if (high - low) < 1.0:  # Minimum bandwidth
            warnings.warn(f"{name} band too narrow")

def calculate_phase_amplitude_coupling(signal_data, fs=2500):
    """Calculate phase-amplitude coupling using Modulation Index (MI) with robust filter design"""
    try:
        if len(signal_data) < 2500:  # Require at least 1 second of data
            return np.nan
        
        # Pre-validate signal quality
        if np.std(signal_data) < 1e-10 or np.all(signal_data == signal_data[0]):
            return np.nan
        
        nyquist = fs / 2
        
        # More conservative frequency bounds for theta (narrower band)
        low_theta = 5 / nyquist    # Start at 5 Hz instead of 4 Hz
        high_theta = 7 / nyquist   # End at 7 Hz instead of 8 Hz
        
        if high_theta >= 0.95 or low_theta <= 0.01:  # More conservative bounds
            return np.nan
        
        # More conservative frequency bounds for gamma
        low_gamma = 35 / nyquist   # Start at 35 Hz instead of 30 Hz
        high_gamma = min(80 / nyquist, 0.9)  # End at 80 Hz and stay well below Nyquist
        
        if high_gamma <= low_gamma or low_gamma >= 0.9:
            return np.nan
        
        # Use lower-order filters for better stability
        filter_order = 3  # Reduced from 4 to 3
        
        # Design theta filter with stability checks
        try:
            b_theta, a_theta = signal.butter(filter_order, [low_theta, high_theta], btype='band')
            
            # Enhanced stability validation
            if len(a_theta) > 1:
                poles = np.roots(a_theta)
                pole_magnitudes = np.abs(poles)
                if np.any(pole_magnitudes >= 0.99):  # More strict threshold
                    # Try with even lower order
                    b_theta, a_theta = signal.butter(2, [low_theta, high_theta], btype='band')
                    poles = np.roots(a_theta)
                    if np.any(np.abs(poles) >= 0.99):
                        return np.nan
            
            # Apply filter with padding to reduce edge effects
            pad_length = min(len(signal_data) // 4, 500)
            theta_signal = signal.filtfilt(b_theta, a_theta, signal_data, padlen=pad_length)
            
        except Exception as e:
            logger.warning(f"Theta filter design failed: {e}")
            return np.nan
        
        # Design gamma filter with stability checks
        try:
            b_gamma, a_gamma = signal.butter(filter_order, [low_gamma, high_gamma], btype='band')
            
            # Enhanced stability validation
            if len(a_gamma) > 1:
                poles = np.roots(a_gamma)
                pole_magnitudes = np.abs(poles)
                if np.any(pole_magnitudes >= 0.99):  # More strict threshold
                    # Try with even lower order
                    b_gamma, a_gamma = signal.butter(2, [low_gamma, high_gamma], btype='band')
                    poles = np.roots(a_gamma)
                    if np.any(np.abs(poles) >= 0.99):
                        return np.nan
            
            # Apply filter with padding
            pad_length = min(len(signal_data) // 4, 500)
            gamma_signal = signal.filtfilt(b_gamma, a_gamma, signal_data, padlen=pad_length)
            
        except Exception as e:
            logger.warning(f"Gamma filter design failed: {e}")
            return np.nan
        
        # Validate filtered signals
        if np.any(np.isnan(theta_signal)) or np.any(np.isnan(gamma_signal)):
            return np.nan
        
        if np.std(theta_signal) < 1e-10 or np.std(gamma_signal) < 1e-10:
            return np.nan
        
        # Extract theta phase using Hilbert transform
        try:
            theta_analytic = hilbert(theta_signal)
            theta_phase = np.angle(theta_analytic)
        except Exception as e:
            logger.warning(f"Theta phase extraction failed: {e}")
            return np.nan
        
        # Extract gamma amplitude using Hilbert transform
        try:
            gamma_analytic = hilbert(gamma_signal)
            gamma_amplitude = np.abs(gamma_analytic)
        except Exception as e:
            logger.warning(f"Gamma amplitude extraction failed: {e}")
            return np.nan
        
        # Validate phase and amplitude arrays
        if np.any(np.isnan(theta_phase)) or np.any(np.isnan(gamma_amplitude)):
            return np.nan
        
        # Calculate Modulation Index using phase binning (Tort et al. 2008)
        n_bins = 18  # Standard number of bins (20° each)
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
        # Calculate mean amplitude in each phase bin
        mean_amplitudes = np.zeros(n_bins)
        valid_bins = 0
        
        for i in range(n_bins):
            phase_mask = (theta_phase >= phase_bins[i]) & (theta_phase < phase_bins[i + 1])
            bin_count = np.sum(phase_mask)
            
            if bin_count > 0:
                mean_amplitudes[i] = np.mean(gamma_amplitude[phase_mask])
                valid_bins += 1
            else:
                mean_amplitudes[i] = 0
        
        # Require minimum number of valid bins
        if valid_bins < n_bins // 2:  # At least half the bins should have data
            return np.nan
        
        # Normalize to create probability distribution P
        total_amplitude = np.sum(mean_amplitudes)
        if total_amplitude <= 0:
            return np.nan
        
        P = mean_amplitudes / total_amplitude
        
        # Remove zero probabilities to avoid log(0)
        P_nonzero = P[P > 1e-10]  # Use small threshold instead of exact zero
        if len(P_nonzero) < 3:  # Need minimum number of non-zero bins
            return np.nan
        
        # Calculate Modulation Index using Kullback-Leibler divergence from uniform distribution
        try:
            uniform_entropy = np.log(n_bins)  # Maximum entropy for uniform distribution
            actual_entropy = -np.sum(P_nonzero * np.log(P_nonzero))  # Shannon entropy
            
            # MI measures deviation from uniform distribution (0 = uniform, 1 = maximum coupling)
            modulation_index = (uniform_entropy - actual_entropy) / uniform_entropy
            
            # Validate result
            if np.isnan(modulation_index) or np.isinf(modulation_index):
                return np.nan
            
            # Clamp to valid range [0, 1]
            modulation_index = np.clip(modulation_index, 0.0, 1.0)
            
            return modulation_index
            
        except Exception as e:
            logger.warning(f"MI calculation failed: {e}")
            return np.nan
        
    except Exception as e:
        logger.warning(f"PAC calculation failed: {e}")
        return np.nan

def extract_features_for_window_size(signal_data, window_size, fs=2500):
    """Extract features from signal data for specific window size - single unified function"""
    signal_data = np.asarray(signal_data, dtype=np.float64)
    signal_data = signal_data[np.isfinite(signal_data)]
    
    # Skip if signal is too short
    if len(signal_data) < window_size:
        return None
    
    # Enhanced artifact detection
    artifacts = detect_signal_artifacts(signal_data, fs)
    if any(artifacts.values()):
        logger.warning(f"Signal artifacts detected: {artifacts}")
    
    # Signal quality validation
    if np.all(signal_data == signal_data[0]):
        warnings.warn("Flat signal detected - may indicate artifact")
    
    if np.std(signal_data) < 1e-10:
        warnings.warn("Very low signal variance - may indicate artifact")
    
    features = {}
    
    # Basic statistical features - RAW VALUES, NO NORMALIZATION (since all windows same size)
    features['min'] = np.min(signal_data)
    features['max'] = np.max(signal_data)
    features['zero_crossings'] = np.sum(np.diff(np.signbit(signal_data)))  # Raw count
    features['energy'] = np.sum(signal_data**2)  # Raw energy
    features['rms'] = np.sqrt(np.mean(signal_data**2))
    
    # Signal variability features
    diff_signal = np.diff(signal_data)
    features['signal_variability'] = np.std(diff_signal)
    features['diff_variance'] = np.var(diff_signal)
    
    # Peak detection features - RAW COUNTS (comparable across same-size windows)
    try:
        threshold = np.mean(signal_data) + 0.5 * np.std(signal_data)
        peaks, _ = find_peaks(signal_data, height=threshold)
        valleys, _ = find_peaks(-signal_data, height=-np.mean(signal_data))
        features['num_peaks'] = len(peaks)  # Raw count
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
    
    # PHASE-AMPLITUDE COUPLING with validation
    features['phase_amplitude_coupling'] = calculate_phase_amplitude_coupling(signal_data, fs)
    
    # Frequency domain features with proper validation
    try:
        nperseg = min(512, len(signal_data)//4)
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg)
        
        # Define frequency bands with validation
        nyquist = fs / 2
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'slow_gamma': (30, 60),
            'fast_gamma': (60, 100),
            'high_freq': (100, min(300, nyquist-1))
        }
        
        # Validate frequency bands
        validate_frequency_bands(fs, bands)
        
        def band_power(freqs, psd, band):
            """Calculate power in specific frequency band with validation"""
            if band[1] >= nyquist:
                return 0
            idx = np.logical_and(freqs >= band[0], freqs <= band[1])
            return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0
        
        # Calculate power in each frequency band
        for band_name, band_range in bands.items():
            features[f'{band_name}_power'] = band_power(freqs, psd, band_range)
        
        # Calculate relative power features
        total_power = np.trapz(psd, freqs)
        if total_power > 0:
            features['high_freq_rel'] = features['high_freq_power'] / total_power
            features['slow_gamma_rel'] = features['slow_gamma_power'] / total_power
        else:
            features['high_freq_rel'] = 0
            features['slow_gamma_rel'] = 0
        
        # Calculate power ratios between bands
        features['fast_gamma_slow_gamma_ratio'] = features['fast_gamma_power'] / (features['slow_gamma_power'] + 1e-10)
        features['gamma_beta_ratio'] = (features['slow_gamma_power'] + features['fast_gamma_power']) / (features['beta_power'] + 1e-10)
        features['fast_slow_gamma_ratio'] = features['fast_gamma_power'] / (features['slow_gamma_power'] + 1e-10)
        features['slow_gamma_theta_ratio'] = features['slow_gamma_power'] / (features['theta_power'] + 1e-10)
        features['fast_gamma_theta_ratio'] = features['fast_gamma_power'] / (features['theta_power'] + 1e-10)
        
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
        # Set default values if frequency analysis fails
        freq_features = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 
                        'slow_gamma_power', 'fast_gamma_power', 'high_freq_power',
                        'high_freq_rel', 'slow_gamma_rel', 'fast_gamma_slow_gamma_ratio',
                        'gamma_beta_ratio', 'fast_slow_gamma_ratio', 'slow_gamma_theta_ratio',
                        'fast_gamma_theta_ratio']
        for feat in freq_features:
            features[feat] = 0
    
    # Handle any remaining NaN or inf values
    for key, value in features.items():
        if np.isnan(value) or np.isinf(value):
            features[key] = 0
    
    return features

def validate_group_separation(cv_splitter, groups, y):
    """Validate that the same animal never appears in both training and test sets within a fold"""
    logger.info("Validating group separation in cross-validation...")
    
    fold_count = 0
    for train_idx, test_idx in cv_splitter.split(np.zeros(len(groups)), y, groups):
        train_animals = set(groups[train_idx])
        test_animals = set(groups[test_idx])
        
        overlap = train_animals.intersection(test_animals)
        if overlap:
            raise ValueError(f"Fold {fold_count}: Animals {overlap} appear in both train and test sets!")
        
        # Check class distribution in each fold
        train_labels = y[train_idx]
        test_labels = y[test_idx]
        
        logger.info(f"Fold {fold_count}:")
        logger.info(f"  Train animals: {sorted(train_animals)} (classes: {np.bincount(train_labels)})")
        logger.info(f"  Test animals: {sorted(test_animals)} (classes: {np.bincount(test_labels)})")
        fold_count += 1
    
    logger.info("* Group separation validation passed - no animal appears in both train and test within any fold")

def create_pl_dataset_with_caching(control, exercised, window_size, overlap):
    """Create dataset with sliding windows, feature extraction, and disk caching"""
    
    logger.info(f"Creating dataset with window_size={window_size}, overlap={overlap}")
    memory_before = monitor_memory_usage()
    
    step_size = int(window_size * (1 - overlap))
    
    X_features = []
    y_labels = []
    groups = []
    animal_features = {}
    
    # Process control animals with caching
    logger.info("Processing control animals...")
    for i, animal_data in enumerate(control):
        animal_id = f'pl_control_{i}'
        
        # Try to load from cache first
        cached_data = cache_manager.load_features(animal_data, window_size, overlap)
        
        if cached_data is not None:
            # Use cached features
            animal_windows = cached_data['features']
            logger.info(f"  Control animal {i+1}/{len(control)} - CACHED ({len(animal_windows)} windows)")
        else:
            # Calculate features
            logger.info(f"  Control animal {i+1}/{len(control)} - COMPUTING...")
            animal_windows = []
            
            # Create sliding windows for each animal
            for start in range(0, len(animal_data) - window_size + 1, step_size):
                window = animal_data[start:start + window_size]
                if len(window) == window_size:
                    feats = extract_features_for_window_size(window, window_size)
                    if feats is not None:
                        animal_windows.append(list(feats.values()))
            
            # Save to cache
            cache_data = {
                'features': animal_windows,
                'window_size': window_size,
                'overlap': overlap,
                'animal_id': animal_id
            }
            cache_manager.save_features(animal_data, window_size, overlap, cache_data)
        
        # Add to dataset
        for window_features in animal_windows:
            X_features.append(window_features)
            y_labels.append(0)  # Control = 0
            groups.append(animal_id)
        
        # Calculate average features per animal
        if animal_windows:
            animal_features[animal_id] = np.mean(animal_windows, axis=0)
    
    # Process exercised animals with caching
    logger.info("Processing exercised animals...")
    for i, animal_data in enumerate(exercised):
        animal_id = f'pl_exercised_{i}'
        
        # Try to load from cache first
        cached_data = cache_manager.load_features(animal_data, window_size, overlap)
        
        if cached_data is not None:
            # Use cached features
            animal_windows = cached_data['features']
            logger.info(f"  Exercised animal {i+1}/{len(exercised)} - CACHED ({len(animal_windows)} windows)")
        else:
            # Calculate features
            logger.info(f"  Exercised animal {i+1}/{len(exercised)} - COMPUTING...")
            animal_windows = []
            
            # Create sliding windows for each animal
            for start in range(0, len(animal_data) - window_size + 1, step_size):
                window = animal_data[start:start + window_size]
                if len(window) == window_size:
                    feats = extract_features_for_window_size(window, window_size)
                    if feats is not None:
                        animal_windows.append(list(feats.values()))
            
            # Save to cache
            cache_data = {
                'features': animal_windows,
                'window_size': window_size,
                'overlap': overlap,
                'animal_id': animal_id
            }
            cache_manager.save_features(animal_data, window_size, overlap, cache_data)
        
        # Add to dataset
        for window_features in animal_windows:
            X_features.append(window_features)
            y_labels.append(1)  # Exercised = 1
            groups.append(animal_id)
        
        # Calculate average features per animal
        if animal_windows:
            animal_features[animal_id] = np.mean(animal_windows, axis=0)
    
    # Get feature names from a sample window
    sample_window = control[0][:window_size] if len(control[0]) >= window_size else control[0]
    sample_features = extract_features_for_window_size(sample_window, window_size)
    feature_names = [
        'min', 'max', 'zero_crossings', 'energy', 'rms', 'signal_variability', 
        'diff_variance', 'num_peaks', 'peak_valley_ratio', 'std_peak_height',
        'phase_amplitude_coupling', 'delta_power', 'theta_power', 'alpha_power',
        'beta_power', 'slow_gamma_power', 'fast_gamma_power', 'high_freq_power',
        'high_freq_rel', 'slow_gamma_rel', 'fast_gamma_slow_gamma_ratio',
        'gamma_beta_ratio', 'fast_slow_gamma_ratio', 'slow_gamma_theta_ratio',
        'fast_gamma_theta_ratio'
    ]    
    
    # Print cache statistics
    cache_stats = cache_manager.get_cache_stats()
    memory_after = monitor_memory_usage()
    logger.info(f"Cache performance: {cache_stats['hits']} hits, {cache_stats['misses']} misses, "
          f"hit rate: {cache_stats['hit_rate']:.2%}")
    logger.info(f"Memory usage: {memory_after:.1f} MB (dif {memory_after - memory_before:.1f} MB)")
    
    return (np.array(X_features), np.array(y_labels), np.array(groups), 
            feature_names, animal_features)

def nested_cross_validation(X, y, groups, feature_names, param_grid, n_outer_folds=5, n_inner_folds=4):
    """Perform nested cross-validation for unbiased performance estimation"""
    logger.info("Starting nested cross-validation...")
    
    # Outer CV for performance estimation - using LeaveTwoAnimalsOut
    outer_cv = LeaveTwoAnimalsOut(shuffle=True, random_state=42)
    
    nested_scores = []
    best_params_list = []
    
    fold_count = 0
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        logger.info(f"Outer fold {fold_count + 1}")
        
        # Split data
        X_train_outer = X[train_idx]
        X_test_outer = X[test_idx]
        y_train_outer = y[train_idx]
        y_test_outer = y[test_idx]
        groups_train_outer = groups[train_idx]
        
        # Inner CV for hyperparameter optimization - use regular StratifiedGroupKFold for inner CV
        inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        
        # Grid search with inner CV
        clf = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Scale features inside the outer fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_outer)
        X_test_scaled = scaler.transform(X_test_outer)
        
        # Fit with inner CV
        clf.fit(X_train_scaled, y_train_outer, groups=groups_train_outer)
        
        # Evaluate on outer test set
        y_pred = clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_outer, y_pred)
        
        nested_scores.append(accuracy)
        best_params_list.append(clf.best_params_)
        
        logger.info(f"  Outer fold {fold_count + 1} accuracy: {accuracy:.4f}")
        logger.info(f"  Best params: {clf.best_params_}")
        fold_count += 1
    
    # Calculate statistics
    mean_score = np.mean(nested_scores)
    std_score = np.std(nested_scores)
    ci_lower, ci_upper = calculate_confidence_intervals(nested_scores)
    
    logger.info(f"Nested CV Results:")
    logger.info(f"  Mean accuracy: {mean_score:.4f} ± {std_score:.4f}")
    logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'mean_score': mean_score,
        'std_score': std_score,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'fold_scores': nested_scores,
        'best_params_per_fold': best_params_list
    }

def analyze_pl_features_with_params(X, y, groups, feature_names, model_params):
    """Perform cross-validation analysis with Leave-Two-Animals-Out cross-validation"""
    
    # Set up cross-validation with proper group separation - using LeaveTwoAnimalsOut
    cv = LeaveTwoAnimalsOut(shuffle=True, random_state=42)
    
    # Validate group separation before proceeding
    validate_group_separation(cv, groups, y)
    
    fold_accuracies = []
    feature_importances = []
    
    # Perform cross-validation with PROPER scaling inside each fold
    fold_count = 0
    for train_idx, test_idx in cv.split(X, y, groups):
        
        # Split data BEFORE any scaling
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # * FIXED: Scale features INSIDE each fold to prevent data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)  # Fit only on training data
        X_test_scaled = scaler.transform(X_test_raw)        # Transform test data using training stats
        
        # Check for multicollinearity in training data
        if fold_count == 0:  # Only check once
            vif_data, high_vif_features = detect_multicollinearity(X_train_scaled, feature_names)
            if high_vif_features:
                logger.warning(f"High multicollinearity detected in features: {high_vif_features}")
        
        # Train model with provided parameters
        model = GradientBoostingClassifier(**model_params)
        model.fit(X_train_scaled, y_train)  # Use scaled training data
        
        # Store feature importance
        feature_importances.append(model.feature_importances_)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)  # Use scaled test data
        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        
        logger.info(f"Fold {fold_count}: Accuracy = {accuracy:.4f}")
        fold_count += 1
    
    # Calculate statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    ci_lower, ci_upper = calculate_confidence_intervals(fold_accuracies)
    
    # Calculate mean feature importance
    mean_feature_importance = np.mean(feature_importances, axis=0)
    importance_df = analyze_feature_importance(
        type('MockModel', (), {'feature_importances_': mean_feature_importance})(),
        feature_names, X
    )
    
    logger.info(f"Mean CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    logger.info(f"Top 5 most important features:")
    for i, row in importance_df.head().iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return mean_accuracy

def objective_with_convergence_monitoring(trial, control, exercised, study):
    """Enhanced objective function with convergence monitoring"""
    
    # Suggest window configuration parameters
    window_size = trial.suggest_int('window_size', 4000, 25000, step=500)
    overlap = trial.suggest_float('overlap', 0.3, 0.8, step=0.05)
    
    # Suggest GradientBoosting hyperparameters with constrained ranges
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 12)  # Constrained for biological signals
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
    min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.1)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 100)
    validation_fraction = trial.suggest_float('validation_fraction', 0.05, 0.2)
    n_iter_no_change = trial.suggest_int('n_iter_no_change', 3, 15)
    tol = trial.suggest_float('tol', 1e-8, 1e-3, log=True)
    loss = trial.suggest_categorical('loss', ['log_loss', 'exponential'])
    criterion = trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error'])
    
    # Create model parameters dictionary
    model_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'min_weight_fraction_leaf': min_weight_fraction_leaf,
        'max_leaf_nodes': max_leaf_nodes,
        'validation_fraction': validation_fraction,
        'n_iter_no_change': n_iter_no_change,
        'tol': tol,
        'loss': loss,
        'criterion': criterion,
        'random_state': 42
    }
    
    try:
        # Create dataset with suggested window configuration and caching
        X, y, groups, feature_names, animal_features = create_pl_dataset_with_caching(
            control, exercised, window_size=window_size, overlap=overlap
        )
        
        # Check if we have enough data
        if len(X) < 100:  # Minimum threshold
            return 0.0  # Poor score for insufficient data
        
        # Analyze with suggested parameters (now with Leave-Two-Animals-Out CV)
        accuracy = analyze_pl_features_with_params(X, y, groups, feature_names, model_params)
        
        # Store additional metrics as user attributes for analysis
        trial.set_user_attr('total_windows', len(X))
        trial.set_user_attr('features_per_window', len(feature_names))
        trial.set_user_attr('control_windows', sum(y==0))
        trial.set_user_attr('exercised_windows', sum(y==1))
        trial.set_user_attr('memory_usage', monitor_memory_usage())
        
        # Check for convergence every 50 trials
        if len(study.trials) % 50 == 0 and len(study.trials) > 100:
            recent_best = max([t.value for t in study.trials[-50:] if t.value is not None])
            overall_best = study.best_value if study.best_value is not None else 0
            if recent_best < overall_best * 1.001:  # Less than 0.1% improvement
                logger.info(f"Potential convergence detected at trial {len(study.trials)}")
        
        return accuracy
        
    except Exception as e:
        logger.error(f"Trial failed with error: {e}")
        return 0.0  # Return poor score for failed trials

def run_optuna_optimization(control, exercised, n_trials=600):
    """Run Optuna optimization for hyperparameters and window configuration with caching"""
    
    logger.info("="*70)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION WITH ENHANCED FEATURES")
    logger.info("="*70)
    logger.info(f"Running {n_trials} trials to optimize:")
    logger.info("- Window size and overlap")
    logger.info("- GradientBoosting hyperparameters")
    logger.info("- Using Leave-Two-Animals-Out cross-validation")
    logger.info("- Features calculated per window size and cached to disk")
    logger.info("- * FIXED: Scaling happens inside each CV fold")
    logger.info("- * Enhanced with convergence monitoring and feature analysis")
    
    # Create study with TPE sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Optimize with progress tracking
    study.optimize(
        lambda trial: objective_with_convergence_monitoring(trial, control, exercised, study), 
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("OPTUNA OPTIMIZATION RESULTS")
    logger.info("="*70)
    
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Get best trial user attributes
    best_trial = study.best_trial
    logger.info(f"\nBest trial details:")
    logger.info(f"  Total windows: {best_trial.user_attrs.get('total_windows', 'N/A')}")
    logger.info(f"  Features per window: {best_trial.user_attrs.get('features_per_window', 'N/A')}")
    logger.info(f"  Control/Exercised windows: {best_trial.user_attrs.get('control_windows', 'N/A')}/{best_trial.user_attrs.get('exercised_windows', 'N/A')}")
    logger.info(f"  Memory usage: {best_trial.user_attrs.get('memory_usage', 'N/A')} MB")
    
    # Print final cache statistics
    final_cache_stats = cache_manager.get_cache_stats()
    logger.info(f"\nFinal cache statistics:")
    logger.info(f"  Cache hits: {final_cache_stats['hits']}")
    logger.info(f"  Cache misses: {final_cache_stats['misses']}")
    logger.info(f"  Cache saves: {final_cache_stats['saves']}")
    logger.info(f"  Hit rate: {final_cache_stats['hit_rate']:.2%}")
    
    return study

def analyze_optimization_results(study):
    """Analyze and visualize optimization results"""
    
    # Create comprehensive analysis
    trials_df = study.trials_dataframe()
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION ANALYSIS")
    logger.info("="*50)
    
    # Top 10 trials
    logger.info("Top 10 trials by accuracy:")
    top_trials = trials_df.nlargest(10, 'value')[['value', 'params_window_size', 'params_overlap', 
                                                  'params_n_estimators', 'params_max_depth', 
                                                  'params_learning_rate']]
    logger.info(top_trials.to_string(index=False))
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        logger.info(f"\nParameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {param}: {imp:.4f}")
    except:
        logger.warning("Could not calculate parameter importance")
    
    # Save results
    trials_df.to_csv('optuna_pl_optimization_results.csv', index=False)
    
    # Create optimization plots
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0,0])
        axes[0,0].set_title('Optimization History')
        
        # Plot 2: Parameter importances
        if len(study.trials) > 10:
            optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0,1])
            axes[0,1].set_title('Parameter Importances')
        
        # Plot 3: Window size vs accuracy
        window_sizes = [trial.params.get('window_size', 0) for trial in study.trials]
        accuracies = [trial.value for trial in study.trials if trial.value is not None]
        if len(window_sizes) == len(accuracies):
            axes[1,0].scatter(window_sizes, accuracies, alpha=0.6)
            axes[1,0].set_xlabel('Window Size')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].set_title('Window Size vs Accuracy')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Overlap vs accuracy
        overlaps = [trial.params.get('overlap', 0) for trial in study.trials]
        if len(overlaps) == len(accuracies):
            axes[1,1].scatter(overlaps, accuracies, alpha=0.6)
            axes[1,1].set_xlabel('Overlap')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_title('Overlap vs Accuracy')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optuna_pl_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"Could not create plots: {e}")

def final_analysis_with_best_params(control, exercised, study):
    """Run final analysis with best parameters found by Optuna"""
    
    logger.info("\n" + "="*70)
    logger.info("FINAL ANALYSIS WITH OPTIMIZED PARAMETERS")
    logger.info("="*70)
    
    best_params = study.best_params
    
    # Extract window parameters
    window_size = best_params['window_size']
    overlap = best_params['overlap']
    
    # Extract model parameters
    model_params = {k: v for k, v in best_params.items() 
                   if k not in ['window_size', 'overlap']}
    model_params['random_state'] = 42
    
    # Create dataset with best window configuration
    X, y, groups, feature_names, animal_features = create_pl_dataset_with_caching(
        control, exercised, window_size=window_size, overlap=overlap
    )
    
    logger.info(f"Optimized Dataset Summary:")
    logger.info(f"- Window size: {window_size} points ({window_size/2500:.2f}s)")
    logger.info(f"- Overlap: {overlap:.3f}")
    logger.info(f"- Total windows: {len(X)}")
    logger.info(f"- Features per window: {len(feature_names)}")
    logger.info(f"- Control/Exercised windows: {sum(y==0)}/{sum(y==1)}")
    logger.info(f"- Unique animals: {len(np.unique(groups))}")
    
    # Run detailed analysis with best parameters
    accuracy = analyze_pl_features_with_params(X, y, groups, feature_names, model_params)
    logger.info(f"\nOptimized accuracy: {accuracy:.4f}")
    
    # Perform nested cross-validation for unbiased estimate
    logger.info("\nPerforming nested cross-validation for unbiased performance estimate...")
    param_grid = {
        'n_estimators': [model_params['n_estimators']],
        'max_depth': [model_params['max_depth']],
        'learning_rate': [model_params['learning_rate']]
    }
    nested_results = nested_cross_validation(X, y, groups, feature_names, param_grid)
    
    # Perform permutation test
    logger.info("\nPerforming permutation test...")
    p_value, perm_accuracies = permutation_test(X, y, groups, feature_names, model_params, n_permutations=500)
    
    # Analyze multicollinearity
    logger.info("\nAnalyzing multicollinearity...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_data, high_vif_features = detect_multicollinearity(X_scaled, feature_names)
    logger.info(f"Features with high VIF (>5): {high_vif_features}")
    
    # Calculate effect sizes between groups
    logger.info("\nCalculating effect sizes...")
    control_accuracies = []
    exercised_accuracies = []
    
    # Get per-animal accuracies for effect size calculation
    unique_animals = np.unique(groups)
    for animal in unique_animals:
        animal_mask = groups == animal
        animal_label = y[animal_mask][0]  # All windows from same animal have same label
        if animal_label == 0:
            control_accuracies.append(np.mean(X[animal_mask], axis=0))
        else:
            exercised_accuracies.append(np.mean(X[animal_mask], axis=0))
    
    if control_accuracies and exercised_accuracies:
        # Calculate Cohen's d for first principal component as overall effect size
        control_pc1 = [np.mean(features) for features in control_accuracies]
        exercised_pc1 = [np.mean(features) for features in exercised_accuracies]
        cohens_d = calculate_cohens_d(exercised_pc1, control_pc1)
        logger.info(f"Cohen's d (overall effect size): {cohens_d:.4f}")
    
    # Save comprehensive results
    optimization_summary = {
        'best_accuracy': study.best_value,
        'best_window_size': window_size,
        'best_overlap': overlap,
        'best_model_params': model_params,
        'nested_cv_results': nested_results,
        'permutation_test': {
            'p_value': p_value,
            'baseline_accuracy': accuracy
        },
        'multicollinearity': {
            'high_vif_features': high_vif_features,
            'vif_data': vif_data.to_dict()
        },
        'effect_size': {
            'cohens_d': cohens_d if 'cohens_d' in locals() else None
        },
        'dataset_summary': {
            'total_windows': len(X),
            'features_per_window': len(feature_names),
            'control_windows': sum(y==0),
            'exercised_windows': sum(y==1),
            'unique_animals': len(np.unique(groups))
        }
    }
    
    # Save to files
    with open('optuna_pl_best_config.json', 'w') as f:
        json.dump(optimization_summary, f, indent=2, default=str)
    
    # Save VIF analysis
    vif_data.to_csv('multicollinearity_analysis_pl.csv', index=False)
    
    logger.info(f"\nOptimization complete!")
    logger.info(f"Files saved:")
    logger.info(f"- optuna_pl_optimization_results.csv")
    logger.info(f"- optuna_pl_best_config.json")
    logger.info(f"- optuna_pl_optimization_analysis.png")
    logger.info(f"- multicollinearity_analysis_pl.csv")
    #logger.info(f"- {logger.handlers[0].baseFilename}")  # Log file name
    # Safer version:
    log_filename = "neural_analysis.log"  # Default fallback
    if logger.handlers:
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                log_filename = handler.baseFilename
                break
    logger.info(f"- {log_filename}")
    
    # Final summary
    logger.info(f"\n" + "="*50)
    logger.info("FINAL SUMMARY")
    logger.info("="*50)
    logger.info(f"Optimized accuracy: {accuracy:.4f}")
    logger.info(f"Nested CV accuracy: {nested_results['mean_score']:.4f} ± {nested_results['std_score']:.4f}")
    logger.info(f"95% CI: [{nested_results['ci_lower']:.4f}, {nested_results['ci_upper']:.4f}]")
    logger.info(f"Permutation test p-value: {p_value:.4f}")
    logger.info(f"Statistical significance: {'Yes' if p_value < 0.05 else 'No'}")
    if 'cohens_d' in locals():
        logger.info(f"Effect size (Cohen's d): {cohens_d:.4f}")
        effect_interpretation = "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
        logger.info(f"Effect size interpretation: {effect_interpretation}")

if __name__ == "__main__":
    # Load data
    #file_path_pl = 'D:/Laboratorio/Registros/Experimental-ejercicio/Schff8-todo.mat'
    file_path_pl = 'D:/Laboratorio/Registros/Experimental-ejercicio/PL6-todo.mat'
    control, exercised = load_pl_data(file_path_pl)
    
    # Run Optuna optimization with increased trials
    study = run_optuna_optimization(control, exercised, n_trials=600)
    
    # Analyze optimization results
    analyze_optimization_results(study)
    
    # Run final analysis with best parameters
    final_analysis_with_best_params(control, exercised, study)
