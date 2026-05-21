import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.signal import find_peaks
import warnings
import logging
import os
from datetime import datetime
from scipy.stats import mannwhitneyu, shapiro, levene

warnings.filterwarnings('ignore')

# --- 1. INITIALIZATION ---

def setup_logging():
    log_filename = f"animal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# --- 2. SIGNAL QUALITY & FEATURES ---

def identify_signal_artifacts(signal_data, fs=2500):
    artifacts = {}
    flat_threshold = np.std(signal_data) * 0.01
    artifacts['flat_segments'] = np.std(signal_data) < flat_threshold
    z_scores = np.abs(stats.zscore(signal_data))
    artifacts['extreme_outliers'] = np.sum(z_scores > 4) / len(signal_data) > 0.05
    return artifacts

def extract_neural_features_no_theta(signal_data, window_size, fs=2500):
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
    #features['phase_amplitude_coupling'] = theta_gamma_phase_amplitude_coupling(signal_data, fs)
    
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

def process_subjects_to_all_windows(subject_list, group_name, window_size=2500):
    """Returns a list of every single valid window found across all subjects."""
    all_windows = []
    for idx, raw_signal in enumerate(subject_list):
        for i in range(0, len(raw_signal) - window_size, window_size):
            window = raw_signal[i : i + window_size]
            feat = extract_neural_features_no_theta(window, window_size)
            if feat:
                feat['subject_id'] = f"{group_name}_{idx}"
                feat['group'] = group_name
                all_windows.append(feat)
    return all_windows

def load_experimental_data(file_path):
    data = scipy.io.loadmat(file_path)
    g_s = data['G']['s'][0, 0]
    control_list = [g_s[0, i].flatten() for i in range(g_s.shape[1]) if len(g_s[0, i].flatten()) > 0]
    exercised_list = [g_s[1, i].flatten() for i in range(g_s.shape[1]) if len(g_s[1, i].flatten()) > 0]
    return control_list, exercised_list

# --- 3. MAIN ---

def main():
    data_file = 'D:/Laboratorio/Registros/Experimental-ejercicio/PL6-todo.mat'
    
    # 1. Extract ALL windows into a single DataFrame
    control_raw, exercised_raw = load_experimental_data(data_file)
    logger.info("Extracting windows and preparing for animal-level aggregation...")
    
    control_windows = process_subjects_to_all_windows(control_raw, 'control')
    exercised_windows = process_subjects_to_all_windows(exercised_raw, 'exercised')
    
    df_windows = pd.DataFrame(control_windows + exercised_windows)

    # --- NEW: ANIMAL-LEVEL AGGREGATION ---
    # We group by subject_id and take the mean of all numeric features
    # 'group' is included in the grouping to preserve it in the resulting index
    df_animal = df_windows.groupby(['subject_id', 'group']).mean().reset_index()
    
    print(f"\nTOTAL ANIMALS ANALYZED: {len(df_animal)}")
    print(df_animal['group'].value_counts())
    print("-" * 30)

    # 2. Stats (Performed on df_animal)
    results = []
    feats = [c for c in df_animal.columns if c not in ['subject_id', 'group']]
    
    print("\n" + "="*60)
    print(f"{'FEATURE':<30} | {'P-VALUE':<10} | {'SIG'}")
    print("="*60)

    for f in feats:
        c_vals = df_animal[df_animal['group'] == 'control'][f].dropna()
        e_vals = df_animal[df_animal['group'] == 'exercised'][f].dropna()
        
        if len(c_vals) < 2 or len(e_vals) < 2: 
            continue
        
        stat, p = mannwhitneyu(c_vals, e_vals)
        
        # Determine significance marker
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        
        results.append({
            'feature': f, 
            'p_value': p, 
            'sig': sig,
            'c_mean': c_vals.mean(), 
            'e_mean': e_vals.mean()
        })

    # Sort results by p-value before printing
    results_df = pd.DataFrame(results)
    
    # 2. Sort and print for the console
    results_sorted = results_df.sort_values('p_value')
    print("\n" + "="*60)
    print(f"{'FEATURE':<30} | {'P-VALUE':<10} | {'SIG'}")
    print("="*60)
    for _, res in results_sorted.iterrows():
        print(f"{res['feature']:<30} | {res['p_value']:.6f}   | {res['sig']}")
    print("="*60)

    # 3. Save to CSV
    results_df.to_csv('animal_level_results.csv', index=False)

    # --- 3. VISUALIZATION ---
    feature_chunks = [feats[i:i + 10] for i in range(0, min(len(feats), 40), 10)]
    
    for grid_idx, chunk in enumerate(feature_chunks):
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        fig.suptitle(f'Animal-Level Mean Distributions - Grid {grid_idx+1}', fontsize=16)
        
        for i, feat in enumerate(chunk):
            ax = axes.flatten()[i]
            sns.boxplot(x='group', y=feat, data=df_animal, ax=ax, palette='Set2', showfliers=False)
            sns.stripplot(x='group', y=feat, data=df_animal, ax=ax, color=".3", alpha=0.6)
            
            # Use results_df (the DataFrame) to get the p-value
            p_val = results_df.loc[results_df['feature'] == feat, 'p_value'].values[0]
            ax.set_title(f"{feat}\n(p={p_val:.4f})", fontsize=9)
            ax.set_xlabel('')
            
        for j in range(len(chunk), 10):
            axes.flatten()[j].axis('off')
            
        plt.savefig(f'animal_feature_grid_{grid_idx+1}.png')
        plt.show()

if __name__ == "__main__":
    main()