import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.signal import find_peaks, hilbert
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import os
from datetime import datetime
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from scipy.stats import mannwhitneyu, shapiro, levene
import json

warnings.filterwarnings('ignore')

def setup_logging():
    """Initialize logging system for statistical analysis"""
    log_filename = f"neural_statistical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def identify_signal_artifacts(signal_data, fs=2500):
    """Detect potential artifacts in neural signal recordings"""
    artifacts = {}
    
    # Check for flat signal segments
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
        if len(signal_data) < 2500:
            return np.nan
        
        if np.std(signal_data) < 1e-10 or np.all(signal_data == signal_data[0]):
            return np.nan
        
        nyquist = fs / 2
        
        # Define frequency bands
        theta_low = 5 / nyquist
        theta_high = 7 / nyquist
        gamma_low = 35 / nyquist
        gamma_high = min(80 / nyquist, 0.9)
        
        if theta_high >= 0.95 or theta_low <= 0.01 or gamma_high <= gamma_low:
            return np.nan
        
        filter_order = 3
        
        # Theta filter
        try:
            b_theta, a_theta = signal.butter(filter_order, [theta_low, theta_high], btype='band')
            if len(a_theta) > 1:
                poles = np.roots(a_theta)
                if np.any(np.abs(poles) >= 0.99):
                    b_theta, a_theta = signal.butter(2, [theta_low, theta_high], btype='band')
                    poles = np.roots(a_theta)
                    if np.any(np.abs(poles) >= 0.99):
                        return np.nan
            
            pad_length = min(len(signal_data) // 4, 500)
            theta_signal = signal.filtfilt(b_theta, a_theta, signal_data, padlen=pad_length)
            
        except Exception:
            return np.nan
        
        # Gamma filter
        try:
            b_gamma, a_gamma = signal.butter(filter_order, [gamma_low, gamma_high], btype='band')
            if len(a_gamma) > 1:
                poles = np.roots(a_gamma)
                if np.any(np.abs(poles) >= 0.99):
                    b_gamma, a_gamma = signal.butter(2, [gamma_low, gamma_high], btype='band')
                    poles = np.roots(a_gamma)
                    if np.any(np.abs(poles) >= 0.99):
                        return np.nan
            
            pad_length = min(len(signal_data) // 4, 500)
            gamma_signal = signal.filtfilt(b_gamma, a_gamma, signal_data, padlen=pad_length)
            
        except Exception:
            return np.nan
        
        # Validate filtered signals
        if (np.any(np.isnan(theta_signal)) or np.any(np.isnan(gamma_signal)) or
            np.std(theta_signal) < 1e-10 or np.std(gamma_signal) < 1e-10):
            return np.nan
        
        # Hilbert transform
        try:
            theta_analytic = hilbert(theta_signal)
            theta_phase = np.angle(theta_analytic)
            
            gamma_analytic = hilbert(gamma_signal)
            gamma_amplitude = np.abs(gamma_analytic)
        except Exception:
            return np.nan
        
        if np.any(np.isnan(theta_phase)) or np.any(np.isnan(gamma_amplitude)):
            return np.nan
        
        # Calculate Modulation Index
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        
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
        
        if valid_bins < n_bins // 2:
            return np.nan
        
        total_amplitude = np.sum(mean_amplitudes)
        if total_amplitude <= 0:
            return np.nan
        
        P = mean_amplitudes / total_amplitude
        P_nonzero = P[P > 1e-10]
        
        if len(P_nonzero) < 3:
            return np.nan
        
        try:
            uniform_entropy = np.log(n_bins)
            actual_entropy = -np.sum(P_nonzero * np.log(P_nonzero))
            
            modulation_index = (uniform_entropy - actual_entropy) / uniform_entropy
            
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
    features['mean'] = np.mean(signal_data)
    features['std'] = np.std(signal_data)
    features['variance'] = np.var(signal_data)
    features['skewness'] = stats.skew(signal_data)
    features['kurtosis'] = stats.kurtosis(signal_data)
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
            features['mean_peak_height'] = np.mean(peak_heights)
            features['std_peak_height'] = np.std(peak_heights)
        else:
            features['mean_peak_height'] = 0
            features['std_peak_height'] = 0
    except:
        features['num_peaks'] = 0
        features['peak_valley_ratio'] = 0
        features['mean_peak_height'] = 0
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
            for band_name in frequency_bands.keys():
                features[f'{band_name}_rel'] = features[f'{band_name}_power'] / total_power
        else:
            for band_name in frequency_bands.keys():
                features[f'{band_name}_rel'] = 0
        
        # Calculate cross-band power ratios
        features['fast_gamma_slow_gamma_ratio'] = features['fast_gamma_power'] / (features['slow_gamma_power'] + 1e-10)
        features['gamma_beta_ratio'] = (features['slow_gamma_power'] + features['fast_gamma_power']) / (features['beta_power'] + 1e-10)
        features['fast_slow_gamma_ratio'] = features['fast_gamma_power'] / (features['slow_gamma_power'] + 1e-10)
        features['slow_gamma_theta_ratio'] = features['slow_gamma_power'] / (features['theta_power'] + 1e-10)
        features['fast_gamma_theta_ratio'] = features['fast_gamma_power'] / (features['theta_power'] + 1e-10)
        features['beta_alpha_ratio'] = features['beta_power'] / (features['alpha_power'] + 1e-10)
        features['theta_delta_ratio'] = features['theta_power'] / (features['delta_power'] + 1e-10)
        
        # Calculate spectral centroid and other spectral features
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_rolloff'] = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
        features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd))
        
    except Exception as e:
        logger.warning(f"Spectral analysis failed: {e}")
        # Set default values for failed spectral features
        spectral_features = ['delta_power', 'theta_power', 'alpha_power', 'beta_power', 
                           'slow_gamma_power', 'fast_gamma_power', 'high_freq_power',
                           'delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel',
                           'slow_gamma_rel', 'fast_gamma_rel', 'high_freq_rel',
                           'fast_gamma_slow_gamma_ratio', 'gamma_beta_ratio', 
                           'fast_slow_gamma_ratio', 'slow_gamma_theta_ratio',
                           'fast_gamma_theta_ratio', 'beta_alpha_ratio', 'theta_delta_ratio',
                           'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth']
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
        raw_data = raw_data[np.isfinite(raw_data)]
        if len(raw_data) > 0:
            control_subjects.append(raw_data)
    
    # Extract exercised group recordings
    for i in range(g_s.shape[1]):
        raw_data = g_s[1, i].flatten()
        raw_data = np.asarray(raw_data, dtype=np.float64)
        raw_data = raw_data[np.isfinite(raw_data)]
        if len(raw_data) > 0:
            exercised_subjects.append(raw_data)
    
    logger.info(f"Loaded data from {len(control_subjects)} control and {len(exercised_subjects)} exercised subjects")
    return control_subjects, exercised_subjects

def calculate_subject_features(subjects_data, group_label, window_size=10000, overlap=0.5):
    """Calculate average features per subject"""
    logger.info(f"Calculating features for {group_label} group...")
    
    step_size = int(window_size * (1 - overlap))
    subject_features = []
    
    for i, subject_data in enumerate(subjects_data):
        logger.info(f"  Processing {group_label} subject {i+1}/{len(subjects_data)}")
        
        # Extract windows from subject's recording
        window_features = []
        for start in range(0, len(subject_data) - window_size + 1, step_size):
            window = subject_data[start:start + window_size]
            if len(window) == window_size:
                features = extract_neural_features(window, window_size)
                if features is not None:
                    window_features.append(features)
        
        if window_features:
            # Average features across windows for this subject
            feature_names = list(window_features[0].keys())
            avg_features = {}
            for feature_name in feature_names:
                values = [w[feature_name] for w in window_features]
                avg_features[feature_name] = np.mean(values)
            
            avg_features['subject_id'] = f'{group_label}_{i}'
            avg_features['group'] = group_label
            subject_features.append(avg_features)
        
        logger.info(f"    Extracted {len(window_features)} windows, averaged to subject-level features")
    
    return subject_features

def perform_statistical_tests(control_data, exercised_data, feature_names, alpha=0.05):
    """Perform comprehensive statistical testing between groups"""
    logger.info("Performing statistical tests between groups...")
    
    results = []
    
    for feature in feature_names:
        if feature in ['subject_id', 'group']:
            continue
            
        control_values = [subj[feature] for subj in control_data if feature in subj and np.isfinite(subj[feature])]
        exercised_values = [subj[feature] for subj in exercised_data if feature in subj and np.isfinite(subj[feature])]
        
        if len(control_values) == 0 or len(exercised_values) == 0:
            logger.warning(f"Skipping {feature}: insufficient data")
            continue
        
        control_values = np.array(control_values)
        exercised_values = np.array(exercised_values)
        
        # Additional filtering for finite values
        control_values = control_values[np.isfinite(control_values)]
        exercised_values = exercised_values[np.isfinite(exercised_values)]
        
        if len(control_values) < 3 or len(exercised_values) < 3:
            logger.warning(f"Skipping {feature}: insufficient valid data points (Control: {len(control_values)}, Exercised: {len(exercised_values)})")
            continue
        
        # Check for zero variance
        if np.var(control_values) == 0 and np.var(exercised_values) == 0:
            logger.warning(f"Skipping {feature}: both groups have zero variance")
            continue
        
        # Basic descriptive statistics
        control_mean = np.mean(control_values)
        control_std = np.std(control_values, ddof=1)
        control_sem = stats.sem(control_values) if len(control_values) > 1 else 0
        
        exercised_mean = np.mean(exercised_values)
        exercised_std = np.std(exercised_values, ddof=1)
        exercised_sem = stats.sem(exercised_values) if len(exercised_values) > 1 else 0
        
        # Test for normality
        control_shapiro_p = np.nan
        exercised_shapiro_p = np.nan
        
        try:
            if len(control_values) >= 3 and len(control_values) <= 5000:
                _, control_shapiro_p = shapiro(control_values)
        except Exception as e:
            logger.debug(f"Shapiro test failed for control {feature}: {e}")
            control_shapiro_p = np.nan
            
        try:
            if len(exercised_values) >= 3 and len(exercised_values) <= 5000:
                _, exercised_shapiro_p = shapiro(exercised_values)
        except Exception as e:
            logger.debug(f"Shapiro test failed for exercised {feature}: {e}")
            exercised_shapiro_p = np.nan
        
        # Test for equal variances
        levene_p = np.nan
        try:
            if np.var(control_values) > 0 and np.var(exercised_values) > 0:
                _, levene_p = levene(control_values, exercised_values)
        except Exception as e:
            logger.debug(f"Levene test failed for {feature}: {e}")
            levene_p = np.nan
        
        # Determine appropriate statistical test
        normal_control = control_shapiro_p > 0.05 if not np.isnan(control_shapiro_p) else False
        normal_exercised = exercised_shapiro_p > 0.05 if not np.isnan(exercised_shapiro_p) else False
        equal_variances = levene_p > 0.05 if not np.isnan(levene_p) else True
        
        # Initialize test results
        t_stat = np.nan
        t_p = np.nan
        test_used = "Failed"
        
        # Perform appropriate statistical test
        try:
            if normal_control and normal_exercised and equal_variances:
                # Independent t-test
                t_stat, t_p = stats.ttest_ind(exercised_values, control_values, equal_var=True)
                test_used = "Independent t-test"
            elif normal_control and normal_exercised and not equal_variances:
                # Welch's t-test
                t_stat, t_p = stats.ttest_ind(exercised_values, control_values, equal_var=False)
                test_used = "Welch's t-test"
            else:
                # Mann-Whitney U test (non-parametric)
                u_stat, t_p = mannwhitneyu(exercised_values, control_values, alternative='two-sided')
                t_stat = u_stat
                test_used = "Mann-Whitney U"
                
            # Validate test results
            if np.isnan(t_stat) or np.isnan(t_p) or np.isinf(t_p):
                logger.warning(f"Invalid test results for {feature}: t_stat={t_stat}, p={t_p}")
                t_stat = np.nan
                t_p = np.nan
                test_used = "Failed"
                
        except Exception as e:
            logger.warning(f"Statistical test failed for {feature}: {e}")
            t_stat = np.nan
            t_p = np.nan
            test_used = "Failed"
        
        # Effect size (Cohen's d) with robust calculation
        cohens_d = 0
        try:
            if len(control_values) > 1 and len(exercised_values) > 1:
                pooled_std = np.sqrt(((len(control_values) - 1) * control_std**2 + 
                                     (len(exercised_values) - 1) * exercised_std**2) / 
                                    (len(control_values) + len(exercised_values) - 2))
                
                if pooled_std > 1e-10:  # Avoid division by zero
                    cohens_d = (exercised_mean - control_mean) / pooled_std
                else:
                    cohens_d = 0
                    
            # Ensure Cohen's d is finite
            if not np.isfinite(cohens_d):
                cohens_d = 0
                
        except Exception as e:
            logger.debug(f"Effect size calculation failed for {feature}: {e}")
            cohens_d = 0
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_magnitude = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_magnitude = "Small"
        elif abs(cohens_d) < 0.8:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
        
        # Confidence interval for mean difference
        diff_mean = exercised_mean - control_mean
        se_diff = np.sqrt(control_sem**2 + exercised_sem**2)
        
        try:
            df = len(control_values) + len(exercised_values) - 2
            if df > 0 and se_diff > 0:
                t_critical = stats.t.ppf(1 - alpha/2, df)
                ci_lower = diff_mean - t_critical * se_diff
                ci_upper = diff_mean + t_critical * se_diff
            else:
                ci_lower = np.nan
                ci_upper = np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan
        
        result = {
            'feature': feature,
            'control_n': len(control_values),
            'control_mean': control_mean,
            'control_std': control_std,
            'control_sem': control_sem,
            'exercised_n': len(exercised_values),
            'exercised_mean': exercised_mean,
            'exercised_std': exercised_std,
            'exercised_sem': exercised_sem,
            'mean_difference': diff_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'test_statistic': t_stat,
            'p_value': t_p,
            'cohens_d': cohens_d,
            'effect_magnitude': effect_magnitude,
            'test_used': test_used,
            'control_normal': normal_control,
            'exercised_normal': normal_exercised,
            'equal_variances': equal_variances,
            'control_shapiro_p': control_shapiro_p,
            'exercised_shapiro_p': exercised_shapiro_p,
            'levene_p': levene_p
        }
        
        results.append(result)
        
        # Debug logging for problematic features
        if np.isnan(t_p):
            logger.info(f"Feature {feature}: FAILED - Control: μ={control_mean:.3f}±{control_std:.3f} (n={len(control_values)}), Exercised: μ={exercised_mean:.3f}±{exercised_std:.3f} (n={len(exercised_values)})")
        else:
            logger.debug(f"Feature {feature}: SUCCESS - p={t_p:.4f}, d={cohens_d:.3f}, test={test_used}")
    
    # Multiple comparisons correction - CRITICAL FIX
    # Only use results with valid p-values
    valid_results = [r for r in results if not np.isnan(r['p_value']) and r['p_value'] is not None]
    invalid_results = [r for r in results if np.isnan(r['p_value']) or r['p_value'] is None]
    
    logger.info(f"Valid statistical tests: {len(valid_results)}, Invalid: {len(invalid_results)}")
    
    if len(valid_results) > 0:
        # Extract p-values from valid results only
        
        print(f"\nDEBUG: Before FDR correction:")
        p_values = [r['p_value'] for r in results]
        print(f"Raw p-values (first 5): {[f'{p:.6f}' for p in p_values[:5]]}")

        # Filter out invalid p-values properly  
        valid_indices = []
        valid_p_values = []

        for i, p_val in enumerate(p_values):
            if not np.isnan(p_val) and p_val is not None and 0 <= p_val <= 1:
                valid_indices.append(i)
                valid_p_values.append(p_val)

        print(f"Valid p-values for correction: {len(valid_p_values)}")

        if len(valid_p_values) > 0:
            try:
                # Apply FDR correction only to valid p-values
                rejected, p_corrected_valid, alpha_sidak, alpha_bonf = multipletests(
                    valid_p_values, alpha=alpha, method='fdr_bh'
                )

                print(f"FDR corrected (first 5): {[f'{p:.6f}' for p in p_corrected_valid[:5]]}")

                # Apply corrections back to original results  
                for i, result_idx in enumerate(valid_indices):
                    results[result_idx]['p_corrected'] = p_corrected_valid[i]
                    results[result_idx]['significant'] = rejected[i]
                    results[result_idx]['significant_uncorrected'] = valid_p_values[i] < alpha

                # Set invalid results
                for i, result in enumerate(results):
                    if i not in valid_indices:
                        result['p_corrected'] = np.nan
                        result['significant'] = False
                        result['significant_uncorrected'] = False

            except Exception as e:
                print(f"FDR correction failed: {e}")
                # Use uncorrected p-values as fallback
                for result in results:
                    result['p_corrected'] = result['p_value'] if not np.isnan(result['p_value']) else np.nan
                    result['significant'] = result['p_value'] < alpha if not np.isnan(result['p_value']) else False
                    result['significant_uncorrected'] = result['p_value'] < alpha if not np.isnan(result['p_value']) else False

            # Set default values for invalid results
            for result in invalid_results:
                result['p_corrected'] = np.nan
                result['significant'] = False
                result['significant_uncorrected'] = False
    
    # Combine all results
    all_results = valid_results + invalid_results
    
    # Sort by effect size (absolute value)
    all_results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
    
    logger.info(f"Statistical testing completed: {len(all_results)} total features analyzed")
    
    return all_results


def create_comprehensive_plots(statistical_results, control_data, exercised_data):
    """Create comprehensive visualization plots"""
    logger.info("Creating comprehensive plots...")
    
    # Prepare data for plotting
    significant_features = [r for r in statistical_results if r['significant']]
    top_features = statistical_results[:20]  # Top 20 by effect size
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Effect sizes plot
    ax1 = plt.subplot(3, 4, 1)
    features_plot = [r['feature'][:20] + '...' if len(r['feature']) > 20 else r['feature'] for r in top_features[:15]]
    effect_sizes = [r['cohens_d'] for r in top_features[:15]]
    colors = ['red' if d < 0 else 'blue' for d in effect_sizes]
    
    bars = ax1.barh(range(len(features_plot)), effect_sizes, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(features_plot)))
    ax1.set_yticklabels(features_plot, fontsize=8)
    ax1.set_xlabel("Cohen's d")
    ax1.set_title("Top 15 Features by Effect Size")
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add effect size magnitude lines
    for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax1.axvline(x=threshold, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=-threshold, color='gray', linestyle=':', alpha=0.5)
    
    # 2. P-values volcano plot
    ax2 = plt.subplot(3, 4, 2)
    effect_sizes_all = [r['cohens_d'] for r in statistical_results]
    p_values_log = [-np.log10(r['p_corrected']) for r in statistical_results]
    significant_mask = [r['significant'] for r in statistical_results]
    
    scatter_colors = ['red' if sig else 'blue' for sig in significant_mask]
    ax2.scatter(effect_sizes_all, p_values_log, c=scatter_colors, alpha=0.6)
    ax2.set_xlabel("Cohen's d")
    ax2.set_ylabel("-log10(p-value corrected)")
    ax2.set_title("Volcano Plot: Effect Size vs Significance")
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.legend()
    
    # 3. Distribution of p-values
    ax3 = plt.subplot(3, 4, 3)
    p_vals = [r['p_value'] for r in statistical_results]
    ax3.hist(p_vals, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('P-value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of P-values')
    ax3.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
    ax3.legend()
    
    # 4. Multiple comparisons visualization
    ax4 = plt.subplot(3, 4, 4)
    p_uncorrected = [r['p_value'] for r in statistical_results]
    p_corrected = [r['p_corrected'] for r in statistical_results]
    ax4.scatter(p_uncorrected, p_corrected, alpha=0.6)
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax4.set_xlabel('Uncorrected p-value')
    ax4.set_ylabel('FDR-corrected p-value')
    ax4.set_title('Multiple Comparisons Correction')
    ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
    ax4.axvline(x=0.05, color='red', linestyle='--', alpha=0.5)
    
    # 5-8. Individual feature comparisons (top 4 significant features)
    plot_positions = [(3, 4, 5), (3, 4, 6), (3, 4, 7), (3, 4, 8)]
    
    for i, (row, col, pos) in enumerate(plot_positions):
        if i < len(significant_features):
            feature_data = significant_features[i]
            feature_name = feature_data['feature']
            
            ax = plt.subplot(row, col, pos)
            
            # Get data for this feature
            control_vals = [subj[feature_name] for subj in control_data if feature_name in subj]
            exercised_vals = [subj[feature_name] for subj in exercised_data if feature_name in subj]
            
            # Create box plot
            data_for_plot = [control_vals, exercised_vals]
            labels = ['Control', 'Exercised']
            
            bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            # Add individual points
            x_control = np.random.normal(1, 0.04, len(control_vals))
            x_exercised = np.random.normal(2, 0.04, len(exercised_vals))
            
            ax.scatter(x_control, control_vals, alpha=0.6, color='blue', s=20)
            ax.scatter(x_exercised, exercised_vals, alpha=0.6, color='red', s=20)
            
            # Add statistics text
            p_val = feature_data['p_corrected']
            cohens_d = feature_data['cohens_d']
            ax.text(0.5, 0.95, f"p={p_val:.4f}\nCohen's d={cohens_d:.3f}", 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            title = feature_name[:25] + '...' if len(feature_name) > 25 else feature_name
            ax.set_title(title, fontsize=10)
            ax.set_ylabel('Value')
    
    # 9. Statistical test methods used
    ax9 = plt.subplot(3, 4, 9)
    test_methods = [r['test_used'] for r in statistical_results]
    test_counts = pd.Series(test_methods).value_counts()
    
    ax9.pie(test_counts.values, labels=test_counts.index, autopct='%1.1f%%', startangle=90)
    ax9.set_title('Statistical Tests Used')
    
    # 10. Effect size distribution
    ax10 = plt.subplot(3, 4, 10)
    effect_sizes_all = [r['cohens_d'] for r in statistical_results]
    ax10.hist(effect_sizes_all, bins=20, alpha=0.7, edgecolor='black')
    ax10.set_xlabel("Cohen's d")
    ax10.set_ylabel('Frequency')
    ax10.set_title('Distribution of Effect Sizes')
    ax10.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Add effect size thresholds
    for threshold, label in [(0.2, 'Small'), (0.5, 'Medium'), (0.8, 'Large')]:
        ax10.axvline(x=threshold, color='red', linestyle=':', alpha=0.7, label=f'{label} ({threshold})')
        ax10.axvline(x=-threshold, color='red', linestyle=':', alpha=0.7)
    
    # 11. Normality test results
    ax11 = plt.subplot(3, 4, 11)
    normality_data = []
    for r in statistical_results:
        if not np.isnan(r['control_shapiro_p']):
            normality_data.append(('Control', r['control_shapiro_p']))
        if not np.isnan(r['exercised_shapiro_p']):
            normality_data.append(('Exercised', r['exercised_shapiro_p']))
    
    if normality_data:
        groups = [d[0] for d in normality_data]
        p_vals_shapiro = [d[1] for d in normality_data]
        
        df_normality = pd.DataFrame({'Group': groups, 'Shapiro_p': p_vals_shapiro})
        sns.violinplot(data=df_normality, x='Group', y='Shapiro_p', ax=ax11)
        ax11.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='p=0.05')
        ax11.set_title('Distribution of Normality Test P-values')
        ax11.set_ylabel('Shapiro-Wilk p-value')
        ax11.legend()
    
    # 12. Power analysis simulation
    ax12 = plt.subplot(3, 4, 12)
    # Calculate achieved power for significant results
    significant_cohens_d = [r['cohens_d'] for r in statistical_results if r['significant']]
    
    if significant_cohens_d:
        # Simulate power for different sample sizes
        sample_sizes = np.arange(5, 50, 2)
        powers = []
        
        mean_effect_size = np.mean([abs(d) for d in significant_cohens_d])
        
        for n in sample_sizes:
            # Approximate power calculation
            se = np.sqrt(2/n)  # Standard error for two-sample test
            t_critical = stats.t.ppf(0.975, 2*n-2)  # Two-tailed test
            power = 1 - stats.t.cdf(t_critical - mean_effect_size/se, 2*n-2)
            powers.append(power)
        
        ax12.plot(sample_sizes, powers, 'b-', linewidth=2)
        ax12.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.8')
        ax12.set_xlabel('Sample Size per Group')
        ax12.set_ylabel('Statistical Power')
        ax12.set_title(f'Power Analysis\n(Mean |Effect Size| = {mean_effect_size:.3f})')
        ax12.grid(True, alpha=0.3)
        ax12.legend()
    
    plt.tight_layout()
    plt.savefig('neural_statistical_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional detailed plots for top significant features
    if len(significant_features) > 4:
        create_detailed_feature_plots(significant_features[4:min(12, len(significant_features))], 
                                    control_data, exercised_data)

def create_detailed_feature_plots(features_to_plot, control_data, exercised_data):
    """Create detailed plots for additional significant features"""
    logger.info("Creating detailed feature plots...")
    
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature_data in enumerate(features_to_plot):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        feature_name = feature_data['feature']
        
        # Get data for this feature
        control_vals = [subj[feature_name] for subj in control_data if feature_name in subj]
        exercised_vals = [subj[feature_name] for subj in exercised_data if feature_name in subj]
        
        # Create combination plot: violin + box + points
        data_combined = pd.DataFrame({
            'Group': ['Control']*len(control_vals) + ['Exercised']*len(exercised_vals),
            'Value': control_vals + exercised_vals
        })
        
        # Violin plot
        parts = ax.violinplot([control_vals, exercised_vals], positions=[1, 2], 
                            widths=0.6, showmeans=True, showmedians=True)
        
        # Color the violin plots
        colors = ['lightblue', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add individual points with jitter
        x_control = np.random.normal(1, 0.04, len(control_vals))
        x_exercised = np.random.normal(2, 0.04, len(exercised_vals))
        
        ax.scatter(x_control, control_vals, alpha=0.8, color='darkblue', s=30, zorder=3)
        ax.scatter(x_exercised, exercised_vals, alpha=0.8, color='darkred', s=30, zorder=3)
        
        # Add mean lines
        ax.hlines(np.mean(control_vals), 0.8, 1.2, colors='darkblue', linewidth=3, label='Control Mean')
        ax.hlines(np.mean(exercised_vals), 1.8, 2.2, colors='darkred', linewidth=3, label='Exercised Mean')
        
        # Customize plot
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Control', 'Exercised'])
        ax.set_ylabel('Value')
        
        title = feature_name.replace('_', ' ').title()[:30]
        if len(feature_name) > 30:
            title += '...'
        ax.set_title(title)
        
        # Add statistics text box
        p_val = feature_data['p_corrected']
        cohens_d = feature_data['cohens_d']
        test_used = feature_data['test_used']
        
        stats_text = f"""p = {p_val:.4f}
Cohen's d = {cohens_d:.3f}
Test: {test_used}
Significant: {'Yes' if feature_data['significant'] else 'No'}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
               fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('neural_statistical_analysis_detailed_features.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(statistical_results, control_data, exercised_data):
    """Create a comprehensive summary report"""
    logger.info("Creating summary report...")
    
    # Basic statistics
    total_features = len(statistical_results)
    significant_features = len([r for r in statistical_results if r['significant']])
    significant_uncorrected = len([r for r in statistical_results if r['significant_uncorrected']])
    
    large_effects = len([r for r in statistical_results if abs(r['cohens_d']) >= 0.8])
    medium_effects = len([r for r in statistical_results if 0.5 <= abs(r['cohens_d']) < 0.8])
    small_effects = len([r for r in statistical_results if 0.2 <= abs(r['cohens_d']) < 0.5])
    
    # Test method distribution
    test_methods = {}
    for r in statistical_results:
        method = r['test_used']
        test_methods[method] = test_methods.get(method, 0) + 1
    
    report = f"""
NEURAL SIGNAL STATISTICAL ANALYSIS REPORT
==========================================

SAMPLE CHARACTERISTICS:
- Control subjects: {len(control_data)}
- Exercised subjects: {len(exercised_data)}
- Total features analyzed: {total_features}

STATISTICAL SIGNIFICANCE:
- Significant after FDR correction: {significant_features} ({significant_features/total_features*100:.1f}%)
- Significant before correction: {significant_uncorrected} ({significant_uncorrected/total_features*100:.1f}%)
- False discovery rate control: Benjamini-Hochberg procedure

EFFECT SIZES (Cohen's d):
- Large effects (|d| >= 0.8): {large_effects} ({large_effects/total_features*100:.1f}%)
- Medium effects (0.5 <= |d| < 0.8): {medium_effects} ({medium_effects/total_features*100:.1f}%)
- Small effects (0.2 <= |d| < 0.5): {small_effects} ({small_effects/total_features*100:.1f}%)

STATISTICAL TESTS USED:
"""
    
    for method, count in test_methods.items():
        report += f"- {method}: {count} ({count/total_features*100:.1f}%)\n"
    
    report += f"""
TOP 10 MOST SIGNIFICANT FEATURES (FDR-corrected):
"""
    
    significant_sorted = [r for r in statistical_results if r['significant']]
    significant_sorted.sort(key=lambda x: x['p_corrected'])
    
    for i, result in enumerate(significant_sorted[:10]):
        direction = "UP" if result['mean_difference'] > 0 else "DOWN"
        report += f"{i+1:2d}. {result['feature']:<30} p={result['p_corrected']:.4f} d={result['cohens_d']:+.3f} {direction}\n"
    
    report += f"""
TOP 10 LARGEST EFFECT SIZES:
"""
    
    for i, result in enumerate(statistical_results[:10]):
        direction = "UP" if result['mean_difference'] > 0 else "DOWN"
        sig_marker = "***" if result['significant'] else ""
        report += f"{i+1:2d}. {result['feature']:<30} d={result['cohens_d']:+.3f} p={result['p_corrected']:.4f} {direction} {sig_marker}\n"
    
    report += f"""
INTERPRETATION NOTES:
- UP indicates exercised group has higher values
- DOWN indicates control group has higher values  
- *** indicates statistical significance after FDR correction
- Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large effect
"""
    
    # Save report with UTF-8 encoding to handle Unicode characters
    with open('neural_statistical_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Summary report saved to neural_statistical_analysis_report.txt")
    return report

def create_all_variables_visualization(control_data, exercised_data, feature_names):
    """Create comprehensive visualization of all variables for both groups"""
    logger.info("Creating visualization for all variables...")
    
    # Filter out non-numeric features
    numeric_features = [f for f in feature_names if f not in ['subject_id', 'group']]
    
    # Calculate grid dimensions
    n_features = len(numeric_features)
    n_cols = 4  # 4 plots per row
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 0:
        return
    
    # Set style
    plt.style.use('default')
    
    for idx, feature in enumerate(numeric_features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Extract data for this feature
        control_vals = [subj.get(feature, np.nan) for subj in control_data]
        exercised_vals = [subj.get(feature, np.nan) for subj in exercised_data]
        
        # Remove NaN values
        control_vals = [v for v in control_vals if not np.isnan(v)]
        exercised_vals = [v for v in exercised_vals if not np.isnan(v)]
        
        if len(control_vals) == 0 or len(exercised_vals) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature.replace('_', ' ').title()[:30])
            continue
        
        # Create box plot
        box_data = [control_vals, exercised_vals]
        box_labels = ['Control', 'Exercised']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       widths=0.6, showmeans=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual data points with jitter
        np.random.seed(42)  # For reproducible jitter
        x_control = np.random.normal(1, 0.05, len(control_vals))
        x_exercised = np.random.normal(2, 0.05, len(exercised_vals))
        
        ax.scatter(x_control, control_vals, alpha=0.8, color='darkblue', 
                  s=40, zorder=3, edgecolors='white', linewidth=0.5)
        ax.scatter(x_exercised, exercised_vals, alpha=0.8, color='darkred', 
                  s=40, zorder=3, edgecolors='white', linewidth=0.5)
        
        # Add mean lines
        control_mean = np.mean(control_vals)
        exercised_mean = np.mean(exercised_vals)
        
        ax.hlines(control_mean, 0.7, 1.3, colors='navy', linewidth=3, 
                 linestyles='solid', alpha=0.8)
        ax.hlines(exercised_mean, 1.7, 2.3, colors='darkred', linewidth=3, 
                 linestyles='solid', alpha=0.8)
        
        # Calculate basic statistics
        control_std = np.std(control_vals, ddof=1) if len(control_vals) > 1 else 0
        exercised_std = np.std(exercised_vals, ddof=1) if len(exercised_vals) > 1 else 0
        
        # Calculate effect size
        if len(control_vals) > 1 and len(exercised_vals) > 1:
            pooled_std = np.sqrt(((len(control_vals) - 1) * control_std**2 + 
                                 (len(exercised_vals) - 1) * exercised_std**2) / 
                                (len(control_vals) + len(exercised_vals) - 2))
            cohens_d = (exercised_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        # Perform statistical test
        try:
            if len(control_vals) >= 3 and len(exercised_vals) >= 3:
                t_stat, p_val = stats.ttest_ind(exercised_vals, control_vals)
                
                # Add statistics text
                stats_text = f"Control: {control_mean:.3f}±{control_std:.3f}\n"
                stats_text += f"Exercise: {exercised_mean:.3f}±{exercised_std:.3f}\n"
                stats_text += f"Cohen's d: {cohens_d:.3f}\n"
                stats_text += f"p = {p_val:.4f}"
                
                # Color the text based on significance and effect size
                if p_val < 0.05:
                    text_color = 'red'
                elif abs(cohens_d) >= 0.5:
                    text_color = 'orange'
                else:
                    text_color = 'black'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       fontsize=8, color=text_color)
        except:
            pass
        
        # Customize plot
        title = feature.replace('_', ' ').title()
        if len(title) > 25:
            title = title[:25] + '...'
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Control', 'Exercised'], fontsize=9)
    
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    for idx in range(n_features, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    # Add overall title
    fig.suptitle('Neural Signal Features: Control vs Exercised Groups', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig('all_variables_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Create summary statistics table
    create_summary_table(control_data, exercised_data, numeric_features)


def create_summary_table(control_data, exercised_data, feature_names):
    """Create a summary statistics table"""
    logger.info("Creating summary statistics table...")
    
    summary_data = []
    
    for feature in feature_names:
        # Extract data
        control_vals = [subj.get(feature, np.nan) for subj in control_data]
        exercised_vals = [subj.get(feature, np.nan) for subj in exercised_data]
        
        # Remove NaN values
        control_vals = [v for v in control_vals if not np.isnan(v)]
        exercised_vals = [v for v in exercised_vals if not np.isnan(v)]
        
        if len(control_vals) == 0 or len(exercised_vals) == 0:
            continue
        
        # Calculate statistics
        control_mean = np.mean(control_vals)
        control_std = np.std(control_vals, ddof=1) if len(control_vals) > 1 else 0
        exercised_mean = np.mean(exercised_vals)
        exercised_std = np.std(exercised_vals, ddof=1) if len(exercised_vals) > 1 else 0
        
        # Effect size
        if len(control_vals) > 1 and len(exercised_vals) > 1:
            pooled_std = np.sqrt(((len(control_vals) - 1) * control_std**2 + 
                                 (len(exercised_vals) - 1) * exercised_std**2) / 
                                (len(control_vals) + len(exercised_vals) - 2))
            cohens_d = (exercised_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        # T-test
        try:
            t_stat, p_val = stats.ttest_ind(exercised_vals, control_vals)
        except:
            t_stat, p_val = np.nan, np.nan
        
        summary_data.append({
            'Feature': feature,
            'Control_Mean': control_mean,
            'Control_Std': control_std,
            'Control_N': len(control_vals),
            'Exercised_Mean': exercised_mean,
            'Exercised_Std': exercised_std,
            'Exercised_N': len(exercised_vals),
            'Mean_Difference': exercised_mean - control_mean,
            'Cohens_D': cohens_d,
            'T_Statistic': t_stat,
            'P_Value': p_val,
            'Effect_Magnitude': 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small' if abs(cohens_d) >= 0.2 else 'Negligible'
        })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by effect size
    summary_df = summary_df.sort_values('Cohens_D', key=abs, ascending=False)
    
    # Save to CSV
    summary_df.to_csv('variables_summary_statistics.csv', index=False)
    
    # Display top 10
    print("\nTOP 10 FEATURES BY EFFECT SIZE:")
    print("="*80)
    print(f"{'Feature':<25} {'Control':<12} {'Exercised':<12} {'Diff':<8} {'Cohen\'s d':<10} {'p-value':<10}")
    print("-"*80)
    
    for _, row in summary_df.head(10).iterrows():
        print(f"{row['Feature']:<25} {row['Control_Mean']:<12.4f} {row['Exercised_Mean']:<12.4f} "
              f"{row['Mean_Difference']:<8.4f} {row['Cohens_D']:<10.3f} {row['P_Value']:<10.4f}")
    
    logger.info("Summary table saved to variables_summary_statistics.csv")
    
    return summary_df


def calculate_all_window_features(subjects_data, group_label, window_size=10000, overlap=0.5):
    """Calculate features for ALL windows (not averaged per subject)"""
    logger.info(f"Calculating features for ALL windows in {group_label} group...")
    
    step_size = int(window_size * (1 - overlap))
    all_window_features = []
    
    for i, subject_data in enumerate(subjects_data):
        logger.info(f"  Processing {group_label} subject {i+1}/{len(subjects_data)}")
        
        # Extract ALL windows from this subject
        window_count = 0
        for start in range(0, len(subject_data) - window_size + 1, step_size):
            window = subject_data[start:start + window_size]
            if len(window) == window_size:
                features = extract_neural_features(window, window_size)
                if features is not None:
                    # Add metadata to track which subject this window came from
                    features['subject_id'] = f'{group_label}_{i}'
                    features['window_id'] = f'{group_label}_{i}_w{window_count}'
                    features['group'] = group_label
                    features['subject_number'] = i
                    all_window_features.append(features)
                    window_count += 1
        
        logger.info(f"    Extracted {window_count} windows from subject {i+1}")
    
    logger.info(f"Total {group_label} windows: {len(all_window_features)}")
    return all_window_features

def perform_window_level_statistical_tests(control_windows, exercised_windows, feature_names, alpha=0.05):
    """Perform statistical tests on ALL windows (matching ML classifier approach)"""
    logger.info("Performing window-level statistical tests...")
    logger.info(f"Control windows: {len(control_windows)}")
    logger.info(f"Exercised windows: {len(exercised_windows)}")
    
    results = []
    
    for feature in feature_names:
        if feature in ['subject_id', 'group', 'window_id', 'subject_number']:
            continue
            
        # Extract values from ALL windows
        control_values = [window[feature] for window in control_windows if feature in window and np.isfinite(window[feature])]
        exercised_values = [window[feature] for window in exercised_windows if feature in window and np.isfinite(window[feature])]
        
        if len(control_values) == 0 or len(exercised_values) == 0:
            logger.warning(f"Skipping {feature}: insufficient data")
            continue
        
        control_values = np.array(control_values)
        exercised_values = np.array(exercised_values)
        
        # Remove non-finite values
        control_values = control_values[np.isfinite(control_values)]
        exercised_values = exercised_values[np.isfinite(exercised_values)]
        
        if len(control_values) < 10 or len(exercised_values) < 10:
            logger.warning(f"Skipping {feature}: insufficient windows (Control: {len(control_values)}, Exercised: {len(exercised_values)})")
            continue
        
        # Basic descriptive statistics
        control_mean = np.mean(control_values)
        control_std = np.std(control_values, ddof=1)
        control_sem = stats.sem(control_values)
        
        exercised_mean = np.mean(exercised_values)
        exercised_std = np.std(exercised_values, ddof=1)
        exercised_sem = stats.sem(exercised_values)
        
        # For large samples, assume normality (Central Limit Theorem) or use non-parametric tests
        # With hundreds/thousands of windows, Mann-Whitney U is often more appropriate
        try:
            # Use Mann-Whitney U test (robust for large samples with different distributions)
            u_stat, p_val = mannwhitneyu(exercised_values, control_values, alternative='two-sided')
            test_used = "Mann-Whitney U"
            test_stat = u_stat
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed for {feature}: {e}")
            try:
                # Fallback to t-test
                t_stat, p_val = stats.ttest_ind(exercised_values, control_values, equal_var=False)
                test_used = "Welch's t-test"
                test_stat = t_stat
            except Exception as e2:
                logger.warning(f"All tests failed for {feature}: {e2}")
                continue
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * control_std**2 + 
                             (len(exercised_values) - 1) * exercised_std**2) / 
                            (len(control_values) + len(exercised_values) - 2))
        
        cohens_d = (exercised_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_magnitude = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_magnitude = "Small"
        elif abs(cohens_d) < 0.8:
            effect_magnitude = "Medium"
        else:
            effect_magnitude = "Large"
        
        # Confidence interval
        diff_mean = exercised_mean - control_mean
        se_diff = np.sqrt(control_sem**2 + exercised_sem**2)
        df = len(control_values) + len(exercised_values) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df) if df > 0 else 1.96
        ci_lower = diff_mean - t_critical * se_diff
        ci_upper = diff_mean + t_critical * se_diff
        
        result = {
            'feature': feature,
            'control_n': len(control_values),
            'control_mean': control_mean,
            'control_std': control_std,
            'control_sem': control_sem,
            'exercised_n': len(exercised_values),
            'exercised_mean': exercised_mean,
            'exercised_std': exercised_std,
            'exercised_sem': exercised_sem,
            'mean_difference': diff_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'test_statistic': test_stat,
            'p_value': p_val,
            'cohens_d': cohens_d,
            'effect_magnitude': effect_magnitude,
            'test_used': test_used
        }
        
        results.append(result)
        logger.debug(f"Feature {feature}: n_control={len(control_values)}, n_exercised={len(exercised_values)}, p={p_val:.6f}, d={cohens_d:.3f}")
    
    # Multiple comparisons correction
    p_values = [r['p_value'] for r in results]
    if len(p_values) > 0:
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        
        for i, result in enumerate(results):
            result['p_corrected'] = p_corrected[i]
            result['significant'] = rejected[i]
            result['significant_uncorrected'] = p_values[i] < alpha
    
    # Sort by effect size
    results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
    
    logger.info(f"Window-level statistical testing completed: {len(results)} features analyzed")
    return results


def create_all_variables_visualization_windows(control_windows, exercised_windows, feature_names):
    """Create comprehensive visualization of all variables for both groups (window-level data)"""
    logger.info("Creating window-level visualization for all variables...")
    
    # Filter out non-numeric features
    numeric_features = [f for f in feature_names if f not in ['subject_id', 'group', 'window_id', 'subject_number']]
    
    # Calculate grid dimensions
    n_features = len(numeric_features)
    n_cols = 4  # 4 plots per row
    n_rows = int(np.ceil(n_features / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 0:
        return
    
    # Set style
    plt.style.use('default')
    
    for idx, feature in enumerate(numeric_features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Extract data for this feature from all windows
        control_vals = [window.get(feature, np.nan) for window in control_windows]
        exercised_vals = [window.get(feature, np.nan) for window in exercised_windows]
        
        # Remove NaN values
        control_vals = [v for v in control_vals if not np.isnan(v)]
        exercised_vals = [v for v in exercised_vals if not np.isnan(v)]
        
        if len(control_vals) == 0 or len(exercised_vals) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature.replace('_', ' ').title()[:30])
            continue
        
        # Create box plot
        box_data = [control_vals, exercised_vals]
        box_labels = ['Control', 'Exercised']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       widths=0.6, showmeans=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add sample of individual data points (subsample for visibility)
        np.random.seed(42)  # For reproducible sampling
        
        # Subsample if we have too many points (for visualization clarity)
        max_points = 200
        if len(control_vals) > max_points:
            control_sample_idx = np.random.choice(len(control_vals), max_points, replace=False)
            control_sample = [control_vals[i] for i in control_sample_idx]
        else:
            control_sample = control_vals
            
        if len(exercised_vals) > max_points:
            exercised_sample_idx = np.random.choice(len(exercised_vals), max_points, replace=False)
            exercised_sample = [exercised_vals[i] for i in exercised_sample_idx]
        else:
            exercised_sample = exercised_vals
        
        # Add jittered points
        x_control = np.random.normal(1, 0.05, len(control_sample))
        x_exercised = np.random.normal(2, 0.05, len(exercised_sample))
        
        ax.scatter(x_control, control_sample, alpha=0.4, color='darkblue', 
                  s=8, zorder=3, edgecolors='none')
        ax.scatter(x_exercised, exercised_sample, alpha=0.4, color='darkred', 
                  s=8, zorder=3, edgecolors='none')
        
        # Add mean lines
        control_mean = np.mean(control_vals)
        exercised_mean = np.mean(exercised_vals)
        
        ax.hlines(control_mean, 0.7, 1.3, colors='navy', linewidth=3, 
                 linestyles='solid', alpha=0.8)
        ax.hlines(exercised_mean, 1.7, 2.3, colors='darkred', linewidth=3, 
                 linestyles='solid', alpha=0.8)
        
        # Calculate basic statistics
        control_std = np.std(control_vals, ddof=1) if len(control_vals) > 1 else 0
        exercised_std = np.std(exercised_vals, ddof=1) if len(exercised_vals) > 1 else 0
        
        # Calculate effect size
        if len(control_vals) > 1 and len(exercised_vals) > 1:
            pooled_std = np.sqrt(((len(control_vals) - 1) * control_std**2 + 
                                 (len(exercised_vals) - 1) * exercised_std**2) / 
                                (len(control_vals) + len(exercised_vals) - 2))
            cohens_d = (exercised_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        else:
            cohens_d = 0
        
        # Perform statistical test
        try:
            if len(control_vals) >= 10 and len(exercised_vals) >= 10:
                # Use Mann-Whitney U for large samples (more robust)
                u_stat, p_val = mannwhitneyu(exercised_vals, control_vals, alternative='two-sided')
                
                # Add statistics text
                stats_text = f"Control: n={len(control_vals)}\n"
                stats_text += f"Exercised: n={len(exercised_vals)}\n"
                stats_text += f"Mean diff: {exercised_mean - control_mean:.3f}\n"
                stats_text += f"Cohen's d: {cohens_d:.3f}\n"
                stats_text += f"p = {p_val:.4f}"
                
                # Color the text based on significance and effect size
                if p_val < 0.05:
                    text_color = 'red'
                elif abs(cohens_d) >= 0.5:
                    text_color = 'orange'
                else:
                    text_color = 'black'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                       fontsize=8, color=text_color)
        except Exception as e:
            logger.debug(f"Statistical test failed for {feature}: {e}")
            pass
        
        # Customize plot
        title = feature.replace('_', ' ').title()
        if len(title) > 25:
            title = title[:25] + '...'
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Control', 'Exercised'], fontsize=9)
    
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    for idx in range(n_features, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    # Add overall title
    fig.suptitle('Neural Signal Features: Control vs Exercised Groups (Window-Level Analysis)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the plot
    plt.savefig('all_variables_comparison_windows.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    logger.info("Window-level visualization saved as 'all_variables_comparison_windows.png'")


def create_window_level_summary_report(statistical_results, n_control_windows, n_exercised_windows):
    """Create a comprehensive summary report for window-level analysis"""
    logger.info("Creating window-level summary report...")
    
    # Basic statistics
    total_features = len(statistical_results)
    significant_features = len([r for r in statistical_results if r.get('significant', False)])
    significant_uncorrected = len([r for r in statistical_results if r.get('significant_uncorrected', False)])
    
    large_effects = len([r for r in statistical_results if abs(r['cohens_d']) >= 0.8])
    medium_effects = len([r for r in statistical_results if 0.5 <= abs(r['cohens_d']) < 0.8])
    small_effects = len([r for r in statistical_results if 0.2 <= abs(r['cohens_d']) < 0.5])
    
    # Test method distribution
    test_methods = {}
    for r in statistical_results:
        method = r.get('test_used', 'Unknown')
        test_methods[method] = test_methods.get(method, 0) + 1
    
    report = f"""
WINDOW-LEVEL NEURAL SIGNAL STATISTICAL ANALYSIS REPORT
======================================================

ANALYSIS APPROACH:
- This analysis matches the ML classifier approach (window-level)
- Each time window treated as independent observation
- Statistical justification for ML feature importance

SAMPLE CHARACTERISTICS:
- Control windows: {n_control_windows}
- Exercised windows: {n_exercised_windows}
- Total windows analyzed: {n_control_windows + n_exercised_windows}
- Total features analyzed: {total_features}

STATISTICAL SIGNIFICANCE:
- Significant after FDR correction: {significant_features} ({significant_features/total_features*100:.1f}%)
- Significant before correction: {significant_uncorrected} ({significant_uncorrected/total_features*100:.1f}%)
- False discovery rate control: Benjamini-Hochberg procedure

EFFECT SIZES (Cohen's d):
- Large effects (|d| >= 0.8): {large_effects} ({large_effects/total_features*100:.1f}%)
- Medium effects (0.5 <= |d| < 0.8): {medium_effects} ({medium_effects/total_features*100:.1f}%)
- Small effects (0.2 <= |d| < 0.5): {small_effects} ({small_effects/total_features*100:.1f}%)

STATISTICAL TESTS USED:
"""
    
    for method, count in test_methods.items():
        report += f"- {method}: {count} ({count/total_features*100:.1f}%)\n"
    
    report += f"""
TOP 10 MOST SIGNIFICANT FEATURES (FDR-corrected):
"""
    
    significant_sorted = [r for r in statistical_results if r.get('significant', False)]
    significant_sorted.sort(key=lambda x: x.get('p_corrected', 1.0))
    
    for i, result in enumerate(significant_sorted[:10]):
        direction = "UP" if result['mean_difference'] > 0 else "DOWN"
        report += f"{i+1:2d}. {result['feature']:<30} p={result.get('p_corrected', 1.0):.4f} d={result['cohens_d']:+.3f} {direction}\n"
    
    report += f"""
TOP 10 LARGEST EFFECT SIZES:
"""
    
    for i, result in enumerate(statistical_results[:10]):
        direction = "UP" if result['mean_difference'] > 0 else "DOWN"
        sig_marker = "***" if result.get('significant', False) else ""
        report += f"{i+1:2d}. {result['feature']:<30} d={result['cohens_d']:+.3f} p={result.get('p_corrected', 1.0):.4f} {direction} {sig_marker}\n"
    
    report += f"""
SAMPLE SIZE COMPARISON:
- Window-level analysis: {n_control_windows + n_exercised_windows} total observations
- Subject-level would be: 16 total observations (8 per group)
- Power increase: {(n_control_windows + n_exercised_windows)/16:.1f}x more data points

ML CLASSIFIER JUSTIFICATION:
- These statistics validate which features are most discriminative
- Window-level analysis matches ML training data structure
- Significant features (p < 0.05 FDR-corrected) show strongest class separation
- Effect sizes indicate practical importance of features for classification

INTERPRETATION NOTES:
- UP indicates exercised group has higher values
- DOWN indicates control group has higher values  
- *** indicates statistical significance after FDR correction
- Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large effect
- Large sample sizes (windows) provide high statistical power
"""
    
    # Save report with UTF-8 encoding
    with open('window_level_statistical_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("Window-level summary report saved to 'window_level_statistical_analysis_report.txt'")
    
    # Also print to console
    print(report)
    
    return report



# Modified main function
def main():
    """Main function for window-level statistical analysis"""
    logger.info("="*70)
    logger.info("WINDOW-LEVEL NEURAL SIGNAL STATISTICAL ANALYSIS")
    logger.info("(Matching ML classifier approach)")
    logger.info("="*70)
    
    # Load configuration
    config_file = 'optuna_pl_best_config.json'
    window_size = 10000
    overlap = 0.5
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        window_size = config.get('best_window_size', window_size)
        overlap = config.get('best_overlap', overlap)
        logger.info(f"Using configuration from {config_file}")
    else:
        logger.info("Using default configuration")
    
    logger.info(f"Window size: {window_size} samples")
    logger.info(f"Window overlap: {overlap:.1%}")
    
    # Load experimental data
    data_file = 'D:/Laboratorio/Registros/Experimental-ejercicio/PL6-todo.mat'
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    
    control_subjects, exercised_subjects = load_experimental_data(data_file)
    
    # Calculate features for ALL windows (not averaged per subject)
    control_windows = calculate_all_window_features(control_subjects, 'control', window_size, overlap)
    exercised_windows = calculate_all_window_features(exercised_subjects, 'exercised', window_size, overlap)
    
    # Get feature names
    if control_windows:
        feature_names = [k for k in control_windows[0].keys() if k not in ['subject_id', 'group', 'window_id', 'subject_number']]
    else:
        logger.error("No features extracted from control group")
        return
    
    logger.info(f"Extracted {len(feature_names)} features per window")
    logger.info(f"Control windows: {len(control_windows)}")
    logger.info(f"Exercised windows: {len(exercised_windows)}")
    
    # Perform window-level statistical tests (matching ML approach)
    statistical_results = perform_window_level_statistical_tests(control_windows, exercised_windows, feature_names)
    
    # Create visualization with all windows
    create_all_variables_visualization_windows(control_windows, exercised_windows, feature_names)
    
    # Save detailed results
    results_df = pd.DataFrame(statistical_results)
    results_df.to_csv('window_level_statistical_analysis_results.csv', index=False)
    
    # Save all window data
    all_window_data = control_windows + exercised_windows
    windows_df = pd.DataFrame(all_window_data)
    windows_df.to_csv('all_window_level_features.csv', index=False)
    
    # Create summary report
    create_window_level_summary_report(statistical_results, len(control_windows), len(exercised_windows))
    
    logger.info("Window-level analysis complete!")



if __name__ == "__main__":
    main()
