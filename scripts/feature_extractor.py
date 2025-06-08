import os
import sys
import time
import multiprocessing
import psutil
from pathlib import Path
from joblib import Parallel, delayed
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
from scipy.signal import welch
from scipy.stats import skew, kurtosis

# Add project root to Python path for absolute imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.signal import normalize, rereference, spectral_entropy, time_filtering
from src.utils.index import ensure_eeg_multiindex

# --- Configuration & Global Constants ---
# Define the number of features globally for consistency
# Time (12): mean, std, var, rms, p2p, skew, kurt, line_length, zcr, hj_mob, hj_cmp, samp_ent
# Freq (5 Abs B_Pow): delta, theta, alpha, beta, gamma
# Freq (5 Rel B_Pow): delta, theta, alpha, beta, gamma
# Freq (5 Spectral): spec_ent, peak_freq, spec_edge_50, spec_edge_95, total_power
# Freq (3 Ratios): alpha/beta, theta/alpha, delta/theta
# Advanced (5): decorr_time, lz_complexity, dfa_alpha, cross_corr_avg, pac_score
# Total = 12 + 5 + 5 + 5 + 3 + 5 = 35 features
NUM_FEATURES_PER_CHANNEL_REF = 35
DEFAULT_EXPECTED_CHANNELS = 19 # Default, can be overridden if known

# --- Core Feature Calculation Helper Functions ---

def _hjorth_parameters(signal: np.ndarray) -> Tuple[float, float]:
    """Calculate Hjorth mobility and complexity parameters."""
    if signal.size < 2: # Need at least 2 points for diff
        return 0.0, 0.0

    dx = np.diff(signal)
    var_x = np.var(signal)
    mobility = 0.0
    complexity = 0.0

    if var_x > 1e-10: # Check variance to avoid division by zero or near-zero
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (var_x + 1e-12)) # Add epsilon for numerical stability

        if var_dx > 1e-10 and dx.size >= 2: # Need at least 2 points in dx for ddx
            ddx = np.diff(dx)
            if ddx.size > 0: # Ensure ddx is not empty (dx had at least 2 points)
                 complexity = np.sqrt(np.var(ddx) / (var_dx + 1e-12))
            # else complexity remains 0.0 (e.g. if dx had only 1 point, ddx is empty)
    return mobility, complexity


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Calculate zero crossing rate."""
    if signal.size == 0:
        return 0.0
    # np.sign returns 0 for 0. To count crossings through true zero,
    # one might need to handle this. Standard ZCR usually counts changes in sign.
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def _sample_entropy(signal: np.ndarray, m: int = 2, r_coeff: float = 0.2) -> float:
    """
    Calculate sample entropy.
    `r_coeff` is the coefficient for the tolerance r (e.g., 0.2 * std(signal)).
    This is a simplified implementation. For critical applications, consider a well-validated library.
    """
    N = len(signal)
    if N <= m : # Need N > m for templates of length m+1
        return 0.0

    std_dev = np.std(signal)
    if std_dev < 1e-10: # Signal is nearly flat, entropy is low or undefined
        return 0.0
    r_tolerance = r_coeff * std_dev

    def _count_matches(current_m_len: int) -> int:
        """Counts pairs of templates of length `current_m_len` that are similar."""
        if N <= current_m_len:
            return 0
        
        # Create templates (subsequences) of length current_m_len
        templates = np.zeros((N - current_m_len + 1, current_m_len))
        for i in range(N - current_m_len + 1):
            templates[i, :] = signal[i : i + current_m_len]

        num_templates = templates.shape[0]
        counts = 0
        # Iterate through unique pairs of templates (i < j)
        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                # Calculate Chebyshev distance (max absolute difference)
                max_abs_diff = np.max(np.abs(templates[i, :] - templates[j, :]))
                if max_abs_diff <= r_tolerance:
                    counts += 1
        return counts

    # Number of template pairs matching for length m
    count_m = _count_matches(m)
    # Number of template pairs matching for length m+1
    count_m1 = _count_matches(m + 1)

    if count_m == 0 or count_m1 == 0:
        # If no matches for m, or no matches for m+1, entropy is typically 0 or undefined.
        # Returning 0 is a common practice. Some definitions might yield -log(0) -> inf.
        # Or, if count_m1 is 0 but count_m is not, it could be -log(small_val/count_m) -> large positive.
        # For stability, if either is zero, result is often taken as 0.
        return 0.0
    else:
        # Standard SampEn definition: -log( (Number of m+1 matches) / (Number of m matches) )
        # The counts here are raw counts of pairs.
        return -np.log(count_m1 / count_m)


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, percentage_power: float = 0.95) -> float:
    """Calculate spectral edge frequency (frequency below which X% of power lies)."""
    if freqs.size == 0 or psd.size == 0 or freqs.size != psd.size:
        return 0.0 # Invalid input
    
    # Ensure psd values are non-negative
    psd_sanitized = np.maximum(psd, 0)
    total_power = np.trapezoid(psd_sanitized, freqs) # Integrate using trapezoidal rule

    if float(total_power) < 1e-12: # Check for zero or near-zero power
        return 0.0 # Or freqs[0] if that's more appropriate

    cumulative_power = np.cumsum(psd_sanitized) * (freqs[1] - freqs[0] if len(freqs) > 1 else 1) # Approximate integration for cumsum
    # A more accurate way for cumulative power using trapz if freqs are not evenly spaced:
    if len(freqs) > 1:
        cumulative_power_values = np.zeros(len(freqs))
        for i in range(1, len(freqs)):
            cumulative_power_values[i] = np.trapezoid(psd_sanitized[:i+1], freqs[:i+1])
        cumulative_power = np.array(cumulative_power_values)
    else: # Single frequency point
        cumulative_power = psd_sanitized * (freqs[0] if freqs.size > 0 else 1)

    threshold_abs_power = total_power * percentage_power
    
    # Find the first index where cumulative power exceeds or equals the threshold
    edge_indices = np.where(cumulative_power >= threshold_abs_power)[0]
    
    if edge_indices.size > 0:
        return float(freqs[edge_indices[0]])
    
    # If threshold is not met (e.g., percentage_power = 1.0 and numerical precision issues)
    # or if all power is concentrated at the very end.
    return float(freqs[-1])


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Find the frequency with maximum power."""
    if freqs.size == 0 or psd.size == 0 or freqs.size != psd.size:
        return 0.0
    if psd.size == 0: # Should be caught by previous, but defensive
        return 0.0
    peak_idx = np.argmax(psd)
    return float(freqs[peak_idx])


# --- Advanced EEG Feature Functions ---

def _decorrelation_time(signal: np.ndarray, fs: int = 250, max_lag_sec: float = 2.0) -> float:
    """
    Calculate decorrelation time - the time lag at which autocorrelation drops to 1/e.
    This measures temporal correlation structure and is sensitive to signal complexity.
    """
    if signal.size < 10:  # Need minimum samples
        return 0.0
    
    # Remove DC component
    signal_centered = signal - np.mean(signal)
    if np.std(signal_centered) < 1e-10:  # Flat signal
        return 0.0
    
    # Calculate maximum lag based on signal length and time constraint
    max_lag_samples = min(len(signal) // 4, int(max_lag_sec * fs))
    if max_lag_samples < 2:
        return 0.0
    
    # Compute autocorrelation using numpy correlate
    autocorr = np.correlate(signal_centered, signal_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
    autocorr = autocorr[:max_lag_samples + 1]  # Limit to max_lag
    
    # Normalize by zero-lag value
    if autocorr[0] <= 0:
        return 0.0
    autocorr_norm = autocorr / autocorr[0]
    
    # Find where autocorrelation drops to 1/e ≈ 0.368
    threshold = 1.0 / np.e
    below_threshold = np.where(autocorr_norm <= threshold)[0]
    
    if below_threshold.size > 0:
        decorr_lag = below_threshold[0]
        return float(decorr_lag / fs)  # Convert to seconds
    else:
        return float(max_lag_sec)  # If never drops below threshold


def _lempel_ziv_complexity(signal: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate Lempel-Ziv complexity - measures signal randomness and complexity.
    Higher values indicate more complex, less predictable signals.
    """
    if signal.size < 2:
        return 0.0
    
    # Binarize signal using median threshold
    median_val = np.median(signal)
    binary_seq = (signal > median_val).astype(int)
    
    # Convert to string for LZ algorithm
    s = ''.join(map(str, binary_seq))
    n = len(s)
    
    if n < 2:
        return 0.0
    
    # Lempel-Ziv complexity calculation
    complexity = 1
    i = 0
    
    while i < n - 1:
        j = 1
        while i + j <= n:
            # Check if current substring exists in previous part
            substring = s[i:i + j]
            if substring in s[:i + j - 1]:
                j += 1
            else:
                break
        complexity += 1
        i += j
    
    if normalize:
        # Normalize by theoretical maximum for random sequence
        max_complexity = n / np.log2(n) if n > 1 else 1
        return float(complexity / max_complexity)
    else:
        return float(complexity)


def _detrended_fluctuation_analysis(signal: np.ndarray, min_scale: int = 4, max_scale: Optional[int] = None) -> float:
    """
    Calculate DFA alpha exponent - characterizes long-range temporal correlations.
    Alpha ~0.5: uncorrelated (white noise)
    Alpha ~1.0: 1/f noise (pink noise)
    Alpha ~1.5: Brownian motion
    """
    if signal.size < 16:  # Need minimum samples for meaningful analysis
        return 0.5  # Return uncorrelated value
    
    # Remove mean and integrate (cumulative sum)
    signal_centered = signal - np.mean(signal)
    integrated_signal = np.cumsum(signal_centered)
    
    n = len(integrated_signal)
    if max_scale is None:
        max_scale = n // 4
    
    # Ensure we have valid scale range
    max_scale = min(max_scale, n // 4)
    if max_scale <= min_scale:
        return 0.5
    
    # Generate scales (box sizes) - logarithmically spaced
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 
                                 num=min(15, max_scale - min_scale + 1)).astype(int))
    
    if len(scales) < 3:  # Need minimum scales for linear fit
        return 0.5
    
    fluctuations = []
    
    for scale in scales:
        if scale >= n:
            continue
            
        # Divide signal into non-overlapping boxes of size 'scale'
        n_boxes = n // scale
        box_fluctuations = []
        
        for i in range(n_boxes):
            start_idx = i * scale
            end_idx = (i + 1) * scale
            box_data = integrated_signal[start_idx:end_idx]
            
            # Fit linear trend to box
            x = np.arange(len(box_data))
            if len(box_data) > 1:
                trend = np.polyfit(x, box_data, 1)
                detrended = box_data - np.polyval(trend, x)
                box_fluctuations.append(np.sqrt(np.mean(detrended**2)))
        
        if box_fluctuations:
            fluctuations.append(np.mean(box_fluctuations))
    
    if len(fluctuations) < 3:
        return 0.5
    
    # Calculate DFA exponent (slope in log-log plot)
    log_scales = np.log10(scales[:len(fluctuations)])
    log_fluctuations = np.log10(np.array(fluctuations))
    
    # Remove any invalid values
    valid_mask = np.isfinite(log_scales) & np.isfinite(log_fluctuations)
    if np.sum(valid_mask) < 3:
        return 0.5
    
    log_scales = log_scales[valid_mask]
    log_fluctuations = log_fluctuations[valid_mask]
    
    # Linear fit to get DFA exponent
    try:
        slope, _ = np.polyfit(log_scales, log_fluctuations, 1)
        return float(np.clip(slope, 0.0, 2.0))  # Clip to reasonable range
    except:
        return 0.5


def _cross_correlation_channels(segment_data: np.ndarray, max_lag: int = 25) -> float:
    """
    Calculate average cross-correlation between all channel pairs.
    Measures spatial connectivity and synchronization between channels.
    """
    if segment_data.ndim != 2 or segment_data.shape[1] < 2:
        return 0.0  # Need at least 2 channels
    
    n_channels = segment_data.shape[1]
    n_samples = segment_data.shape[0]
    
    if n_samples < max_lag * 2:
        max_lag = min(max_lag, n_samples // 4)
    
    if max_lag < 1:
        return 0.0
    
    correlations = []
    
    # Calculate cross-correlation for all unique channel pairs
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            chan1 = segment_data[:, i]
            chan2 = segment_data[:, j]
            
            # Remove DC component
            chan1_centered = chan1 - np.mean(chan1)
            chan2_centered = chan2 - np.mean(chan2)
            
            # Skip if either channel has no variance
            if np.std(chan1_centered) < 1e-10 or np.std(chan2_centered) < 1e-10:
                continue
            
            # Calculate cross-correlation
            cross_corr = np.correlate(chan1_centered, chan2_centered, mode='full')
            
            # Normalize by signal norms
            norm_factor = np.sqrt(np.sum(chan1_centered**2) * np.sum(chan2_centered**2))
            if norm_factor > 1e-10:
                cross_corr = cross_corr / norm_factor
            
            # Extract correlation around zero lag
            center_idx = len(cross_corr) // 2
            start_idx = max(0, center_idx - max_lag)
            end_idx = min(len(cross_corr), center_idx + max_lag + 1)
            
            correlation_window = cross_corr[start_idx:end_idx]
            
            # Take maximum absolute correlation within the window
            if correlation_window.size > 0:
                max_corr = np.max(np.abs(correlation_window))
                correlations.append(max_corr)
    
    if correlations:
        return float(np.mean(correlations))
    else:
        return 0.0


def _phase_amplitude_coupling(signal: np.ndarray, fs: int = 250, 
                            low_freq_band: Tuple[float, float] = (4, 8),
                            high_freq_band: Tuple[float, float] = (30, 50)) -> float:
    """
    Calculate phase-amplitude coupling (PAC) - measures coupling between 
    low-frequency phase and high-frequency amplitude.
    This is important for understanding cross-frequency interactions in EEG.
    """
    if signal.size < fs:  # Need at least 1 second of data
        return 0.0
    
    try:
        from scipy.signal import hilbert, sosfiltfilt, butter
        import numpy as np
    except ImportError:
        return 0.0  # If scipy not available, return default
    
    # Design bandpass filters
    nyquist = fs / 2
    
    # Low frequency filter (for phase)
    if low_freq_band[1] >= nyquist:
        return 0.0
    low_sos = butter(4, [low_freq_band[0]/nyquist, low_freq_band[1]/nyquist], 
                     btype='band', output='sos')
    
    # High frequency filter (for amplitude)
    if high_freq_band[1] >= nyquist:
        high_freq_band = (high_freq_band[0], nyquist - 1)
    if high_freq_band[0] >= high_freq_band[1]:
        return 0.0
    high_sos = butter(4, [high_freq_band[0]/nyquist, high_freq_band[1]/nyquist], 
                      btype='band', output='sos')
    
    try:
        # Filter signals using sosfiltfilt
        low_freq_signal = sosfiltfilt(low_sos, signal)
        high_freq_signal = sosfiltfilt(high_sos, signal)
        
        # Extract phase from low frequency and amplitude from high frequency
        low_analytic_signal = hilbert(low_freq_signal)
        high_analytic_signal = hilbert(high_freq_signal)
        
        # Ensure we have arrays, not tuples
        low_analytic_array = np.asarray(low_analytic_signal)
        high_analytic_array = np.asarray(high_analytic_signal)
        
        low_phase = np.angle(low_analytic_array)
        high_amplitude = np.abs(high_analytic_array)
        
        # Calculate mean vector length (phase-amplitude coupling strength)
        # This is a simplified PAC measure
        complex_coupling = high_amplitude * np.exp(1j * low_phase)
        mean_vector_length = np.abs(np.mean(complex_coupling)) / np.mean(high_amplitude)
        
        return float(np.clip(mean_vector_length, 0.0, 1.0))
        
    except Exception:
        return 0.0

# --- Main Feature Extraction per Channel ---

def _extract_channel_features(
    channel_signal: np.ndarray,
    fs: int,
    bands: Dict[str, Tuple[float, float]],
    all_channels_data: Optional[np.ndarray] = None
) -> List[float]:
    """
    Extracts a comprehensive set of EEG features from a single channel,
    including advanced features for seizure detection.
    """
    # Need enough data for meaningful analysis, e.g., at least one full cycle of lowest band or for Welch window
    min_len_for_analysis = max(fs / bands.get("delta", (0.5,4))[0], fs * 0.5) # e.g. 0.5 sec for Welch
    if len(channel_signal) < min_len_for_analysis:
        return [0.0] * NUM_FEATURES_PER_CHANNEL_REF

    # === TIME DOMAIN FEATURES ===
    mean_val = np.mean(channel_signal)
    std_val = np.std(channel_signal)
    variance = std_val**2 # More direct than np.var if std_val is already computed
    rms_val = np.sqrt(np.mean(channel_signal**2))
    peak_to_peak = np.ptp(channel_signal) # np.max(channel_signal) - np.min(channel_signal)

    skewness = skew(channel_signal) if len(channel_signal) > 1 else 0.0
    kurt_val = kurtosis(channel_signal, fisher=True) if len(channel_signal) > 1 else 0.0 # Fisher's definition (normal ==> 0)

    line_length = np.sum(np.abs(np.diff(channel_signal))) if len(channel_signal) > 1 else 0.0
    zero_cross_rate = _zero_crossing_rate(channel_signal)
    hj_mob, hj_cmp = _hjorth_parameters(channel_signal)
    sample_ent = _sample_entropy(channel_signal) # Can be computationally intensive

    # === FREQUENCY DOMAIN FEATURES ===
    nperseg_welch = min(len(channel_signal), fs * 2) # Max 2-second window for Welch, or signal length if shorter

    abs_band_powers = {name: 0.0 for name in bands}
    rel_band_powers = {name: 0.0 for name in bands}
    spec_ent_val = 0.0
    peak_freq = 0.0
    spec_edge_50 = 0.0 # Median power frequency
    spec_edge_95 = 0.0 # 95% power frequency
    total_power_val = 0.0
    alpha_beta_ratio, theta_alpha_ratio, delta_theta_ratio = 0.0, 0.0, 0.0

    if len(channel_signal) >= nperseg_welch and nperseg_welch > 0:
        try:
            freqs, psd = welch(channel_signal, fs=fs, nperseg=nperseg_welch, scaling='density', average='mean')
            
            if freqs.size > 0 and psd.size > 0:
                psd = np.maximum(psd, 0) # Ensure non-negative PSD
                total_power_val = np.trapezoid(psd, freqs)
                if total_power_val < 1e-12: total_power_val = 1e-12 # Avoid division by zero later

                for name, (lo, hi) in bands.items():
                    idx_band = (freqs >= lo) & (freqs < hi)
                    if np.any(idx_band):
                        abs_band_powers[name] = np.trapezoid(psd[idx_band], freqs[idx_band])
                        rel_band_powers[name] = abs_band_powers[name] / total_power_val
                
                psd_normalized = psd / total_power_val # Normalize for entropy
                # Use only positive, non-zero parts of PSD for entropy calculation
                psd_nz = psd_normalized[psd_normalized > 1e-12]
                if psd_nz.size > 0:
                    spec_ent_val = spectral_entropy(psd_nz, normalize=True)

                peak_freq = _peak_frequency(freqs, psd)
                spec_edge_50 = _spectral_edge_frequency(freqs, psd, 0.50)
                spec_edge_95 = _spectral_edge_frequency(freqs, psd, 0.95)

                if abs_band_powers.get("beta", 0.0) > 1e-12:
                    alpha_beta_ratio = abs_band_powers.get("alpha", 0.0) / abs_band_powers.get("beta", 0.0)
                if abs_band_powers.get("alpha", 0.0) > 1e-12:
                    theta_alpha_ratio = abs_band_powers.get("theta", 0.0) / abs_band_powers.get("alpha", 0.0)
                if abs_band_powers.get("theta", 0.0) > 1e-12:
                    delta_theta_ratio = abs_band_powers.get("delta", 0.0) / abs_band_powers.get("theta", 0.0)
        except (ValueError, FloatingPointError, IndexError) as e:
            print(f"Debug: Welch PSD calculation or subsequent freq feature failed: {e}")
            pass

    # === ADVANCED EEG FEATURES ===
    # 1. Decorrelation time
    decorr_time = _decorrelation_time(channel_signal, fs)
    
    # 2. Lempel-Ziv complexity
    lz_complexity = _lempel_ziv_complexity(channel_signal)
    
    # 3. Detrended fluctuation analysis
    dfa_alpha = _detrended_fluctuation_analysis(channel_signal)
    
    # 4. Cross-correlation between channels (if multi-channel data available)
    cross_corr_avg = 0.0
    if all_channels_data is not None and all_channels_data.ndim == 2:
        cross_corr_avg = _cross_correlation_channels(all_channels_data)
    
    # 5. Phase-amplitude coupling
    pac_score = _phase_amplitude_coupling(channel_signal, fs)

    features = [
        # Time domain (12)
        mean_val, std_val, variance, rms_val, peak_to_peak,
        skewness, kurt_val, line_length, zero_cross_rate,
        hj_mob, hj_cmp, sample_ent,
        # Absolute band powers (5)
        abs_band_powers.get("delta", 0.0), abs_band_powers.get("theta", 0.0),
        abs_band_powers.get("alpha", 0.0), abs_band_powers.get("beta", 0.0),
        abs_band_powers.get("gamma", 0.0),
        # Relative band powers (5)
        rel_band_powers.get("delta", 0.0), rel_band_powers.get("theta", 0.0),
        rel_band_powers.get("alpha", 0.0), rel_band_powers.get("beta", 0.0),
        rel_band_powers.get("gamma", 0.0),
        # Spectral features (5)
        spec_ent_val, peak_freq, spec_edge_50, spec_edge_95, total_power_val,
        # Band ratios (3)
        alpha_beta_ratio, theta_alpha_ratio, delta_theta_ratio,
        # Advanced features (5)
        decorr_time, lz_complexity, dfa_alpha, cross_corr_avg, pac_score,
    ]
    
    if len(features) != NUM_FEATURES_PER_CHANNEL_REF:
        # This case should ideally not be hit if the logic is correct.
        # Fallback to ensure consistent feature vector length.
        padded_features = [0.0] * NUM_FEATURES_PER_CHANNEL_REF
        # Copy the computed features, truncating or padding as necessary
        copy_len = min(len(features), NUM_FEATURES_PER_CHANNEL_REF)
        padded_features[:copy_len] = features[:copy_len]
        return padded_features

    return features


def _extract_segment_features(segment_signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """Extract features from all channels in a segment using the enhanced feature set."""
    all_channel_features: List[float] = []
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
    
    actual_num_channels = segment_signal.shape[1]
    if actual_num_channels == 0: # Should not happen if segment is valid
        return np.array([0.0] * (DEFAULT_EXPECTED_CHANNELS * NUM_FEATURES_PER_CHANNEL_REF))

    for ch_idx in range(actual_num_channels):
        ch_signal_data = segment_signal[:, ch_idx]
        # Pass the full multi-channel data for cross-correlation calculation
        ch_feats = _extract_channel_features(ch_signal_data, fs, bands, all_channels_data=segment_signal)
        all_channel_features.extend(ch_feats)

    features_array = np.asarray(all_channel_features, dtype=float)
    
    # If the number of actual channels differs from a fixed expectation,
    # the total length of features_array will vary. This is handled later during DataFrame creation.
    # For now, just ensure NaNs/Infs are handled.
    return np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)


def process_session_for_features(
    session_group: Tuple[Any, pd.DataFrame],
    base_signal_path: Path,
    f_s: int = 250,
    num_channels_expected: int = DEFAULT_EXPECTED_CHANNELS, # Used for empty_features if signal load fails
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[Any]]:
    group_name, session_df = session_group
    job_start_time = time.time()
    process_id = os.getpid()
    
    num_features_per_channel = NUM_FEATURES_PER_CHANNEL_REF

    status_msg = f"🔄 [PID:{process_id}] Starting: {group_name} ({len(session_df)} segs) at {time.strftime('%H:%M:%S')}"
    if verbose: print(status_msg)

    last_status_update_time = time.time()
    def _log_status_update(stage_msg, force_print=False):
        nonlocal last_status_update_time
        current_time = time.time()
        if force_print or (current_time - last_status_update_time) > 30: # Update every 30s or if forced
            mem_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            elapsed_total = current_time - job_start_time
            print(f"    🕒 [PID:{process_id}] {group_name} - {stage_msg} - Elap: {elapsed_total:.1f}s - Mem: {mem_usage_mb:.1f}MB")
            last_status_update_time = current_time

    # Pre-allocate empty features based on *expected* channels.
    empty_features_alloc = np.zeros(num_channels_expected * num_features_per_channel, dtype=float)

    features_list_for_session: List[np.ndarray] = []
    all_original_indices_for_session: List[Any] = []

    if 'signals_path' not in session_df.columns or session_df.empty:
        if verbose: print(f"    ⚠️ No 'signals_path' or empty session for {group_name}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features_alloc.copy())
            all_original_indices_for_session.append(original_idx)
        if verbose: print(f"    ✅ {group_name} done (empty session) in {time.time() - job_start_time:.2f}s")
        return features_list_for_session, all_original_indices_for_session

    relative_signal_file = session_df['signals_path'].iloc[0]
    full_signal_path = base_signal_path / relative_signal_file
    
    if verbose: print(f"    📁 Loading: {full_signal_path}")

    if not full_signal_path.exists():
        if verbose: print(f"    ❌ File not found: {full_signal_path}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features_alloc.copy())
            all_original_indices_for_session.append(original_idx)
        if verbose: print(f"    ✅ {group_name} done (file not found) in {time.time() - job_start_time:.2f}s")
        return features_list_for_session, all_original_indices_for_session

    try:
        _log_status_update("Reading parquet", force_print=verbose)
        signal_load_start = time.time()
        session_eeg_data_df = pd.read_parquet(full_signal_path)
        session_eeg_values = session_eeg_data_df.values # Shape: (time_points, actual_num_channels)
        actual_num_channels_in_file = session_eeg_values.shape[1]
        if verbose: print(f"    ✅ Loaded signal {session_eeg_values.shape} in {time.time() - signal_load_start:.2f}s")
    except Exception as e:
        print(f"    ❌ Error loading {full_signal_path} for {group_name}: {e}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features_alloc.copy()) # Use pre-allocated
            all_original_indices_for_session.append(original_idx)
        if verbose: print(f"    ✅ {group_name} done (load error) in {time.time() - job_start_time:.2f}s")
        return features_list_for_session, all_original_indices_for_session

    # If actual channels differ from expected, create a specific empty_features for this session
    current_session_empty_features = empty_features_alloc
    if actual_num_channels_in_file != num_channels_expected:
        if verbose:
            print(f"    ℹ️ Channel mismatch for {group_name}: Expected {num_channels_expected}, Got {actual_num_channels_in_file}. Adjusting empty features.")
        current_session_empty_features = np.zeros(actual_num_channels_in_file * num_features_per_channel, dtype=float)

    _log_status_update("Filtering, Rereferencing & Normalization", force_print=verbose)
    filter_proc_start = time.time()
    
    # signal time-filtering
    bp_filter = signal.butter(4, (0.5, 50), btype="bandpass", output="sos", fs=f_s) # bandpass filter for EEG signals: selects frequencies between 0.5 and 50Hz
    notch_filter = signal.tf2sos(*signal.iirnotch(w0=60, Q=30, fs=f_s)) # notch filter to remove 60 Hz noise (fixed frequency)
    processed_signal_data = time_filtering(session_eeg_values, bp_filter, notch_filter)
    # signal rereferencing
    processed_signal_data = rereference(processed_signal_data)
    # signal normalization
    processed_signal_data = normalize(processed_signal_data)

    if verbose: print(f"    ✅ Filtering, CAR Rereferencing & Normalization done in {time.time() - filter_proc_start:.2f}s")

    processed_segment_count = 0
    valid_segment_features_count = 0
    failed_segment_extraction_count = 0

    for original_idx, segment_meta_row in session_df.iterrows():
        processed_segment_count += 1
        if verbose and (processed_segment_count <= 3 or processed_segment_count % 50 == 0):
             _log_status_update(f"Segment {processed_segment_count}/{len(session_df)}", force_print=True)
        
        try:
            start_sample = int(segment_meta_row["start_time"] * f_s)
            end_sample = int(segment_meta_row["end_time"] * f_s)
            signal_max_len = processed_signal_data.shape[0]
            
            is_valid_segment_time = (0 <= start_sample < signal_max_len and start_sample < end_sample)
            if not is_valid_segment_time:
                if verbose and processed_segment_count <=3: print(f"        Segment {processed_segment_count} Invalid time: start={start_sample}, end={end_sample}, max_len={signal_max_len}")
                features_list_for_session.append(current_session_empty_features.copy())
                all_original_indices_for_session.append(original_idx)
                failed_segment_extraction_count += 1
                continue
            
            end_sample = min(end_sample, signal_max_len) # Clip end_sample to signal length
            eeg_segment_data = processed_signal_data[start_sample:end_sample, :]

            # Min length for feature extraction (e.g., 0.1 seconds)
            if eeg_segment_data.shape[0] < max(10, f_s // 10):
                if verbose and processed_segment_count <=3: print(f"        Segment {processed_segment_count} too short: {eeg_segment_data.shape[0]} samples")
                features_list_for_session.append(current_session_empty_features.copy())
                all_original_indices_for_session.append(original_idx)
                failed_segment_extraction_count += 1
                continue
            
            segment_features = _extract_segment_features(eeg_segment_data, fs=f_s)
            features_list_for_session.append(segment_features)
            all_original_indices_for_session.append(original_idx)
            valid_segment_features_count += 1

        except Exception as e_seg:
            features_list_for_session.append(current_session_empty_features.copy())
            all_original_indices_for_session.append(original_idx)
            failed_segment_extraction_count += 1
            if verbose:
                print(f"        ❌ Error processing segment {processed_segment_count} (orig_idx {original_idx}) for {group_name}: {e_seg}")
            continue
    
    total_job_duration = time.time() - job_start_time
    if verbose:
        print(f"    ✅ [PID:{process_id}] {group_name} finished in {total_job_duration:.2f}s.")
        print(f"        Segments: Total={processed_segment_count}, ValidFeats={valid_segment_features_count}, FailedExtr={failed_segment_extraction_count}")
    elif processed_segment_count > 0 : # Brief summary if not verbose
        print(f"    [PID:{process_id}] {group_name}: {valid_segment_features_count}/{processed_segment_count} valid feats ({total_job_duration:.2f}s)")

    return features_list_for_session, all_original_indices_for_session


# --- Main Execution Logic ---
IS_SCITAS = True # User specific: Set to True if running on SCITAS environment
CPU_COUNT = multiprocessing.cpu_count()

def main(verbose: bool = False, test_mode: bool = False, max_workers: Optional[int] = None):
    overall_pipeline_start_time = time.time()
    print(f"\n🧠 EEG FEATURE EXTRACTION PIPELINE (Full Refactored Script)")
    print(f"{'='*60}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 System info: {CPU_COUNT} CPU cores available")
    
    features_per_channel = NUM_FEATURES_PER_CHANNEL_REF
    expected_channels = DEFAULT_EXPECTED_CHANNELS # For printouts and default empty features
    print(f"📈 Features per channel: {features_per_channel}, Expected channels: {expected_channels}")

    if max_workers is None:
        max_workers = max(1, CPU_COUNT - 1 if CPU_COUNT > 1 else 1)
    max_workers = min(max_workers, CPU_COUNT, 16) # Cap workers, e.g., at 16 or a reasonable number
    print(f"🔧 Config: verbose={verbose}, test_mode={test_mode}, max_workers={max_workers}")
    
    # Define data paths
    # Adjust DATA_ROOT if not running from the project root or if data is elsewhere
    project_root = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
    LOCAL_DATA_ROOT = project_root / "data"
    
    # SCITAS path needs to be configured by the user if IS_SCITAS is True
    SCITAS_DATA_ROOT = Path("/home/ogut/data") # Placeholder: User should configure this
    DATA_ROOT = SCITAS_DATA_ROOT if IS_SCITAS else LOCAL_DATA_ROOT
    
    print(f"🔍 Using DATA_ROOT: {DATA_ROOT.resolve()}")
    (DATA_ROOT / "extracted_features").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    
    if test_mode: print(f"\n🧪 TEST MODE ENABLED - Processing limited data samples.")
    if verbose:
        print(f"\n📊 REFINED EEG FEATURE EXTRACTION DETAILS:")
        print(f"   Features per channel: {features_per_channel}")
        print(f"   Total expected features per segment (for {expected_channels} channels): {expected_channels * features_per_channel}")
        print(f"   Parallel processing with up to {max_workers} workers.")
        print(f"{'='*50}\n")

    try:
        clips_tr_full_df = pd.read_parquet(DATA_ROOT / "train" / "segments.parquet")
        clips_te_full_df = pd.read_parquet(DATA_ROOT / "test" / "segments.parquet")
        if verbose: print(f"✅ Loaded segment metadata: Train {clips_tr_full_df.shape}, Test {clips_te_full_df.shape}")
    except FileNotFoundError as e:
        print(f"❌ CRITICAL Error: Segments.parquet file not found. Path: {e.filename}. Ensure data is in {DATA_ROOT}/train and {DATA_ROOT}/test.")
        sys.exit(1)

    clips_tr_full_df = ensure_eeg_multiindex(clips_tr_full_df, id_col_name='id')
    clips_te_full_df = ensure_eeg_multiindex(clips_te_full_df, id_col_name='id')

    train_segments_for_labels_df = clips_tr_full_df[~clips_tr_full_df['label'].isna()].copy()
    test_segments_df = clips_te_full_df.copy()
    if verbose:
        print(f"   Training segments (valid labels): {train_segments_for_labels_df.shape}")
        print(f"   Test segments: {test_segments_df.shape}")

    SAMPLING_RATE = 250
    BANDPASS_FREQS = (0.5, 50.0) # Consistent with Gamma band upper limit
    NOTCH_FILTER_HZ = 60.0 # Or 50.0 depending on region of data origin
    NOTCH_Q = 30.0

    print(f"\n🔧 Designing filters: BP={BANDPASS_FREQS}Hz, Notch={NOTCH_FILTER_HZ}Hz (Q={NOTCH_Q}), FS={SAMPLING_RATE}Hz")
    
    grouping_cols = ["patient", "session"] # For grouping segments by session signal file

    # --- Training Set Feature Extraction ---
    print("\n⏳ Processing Training Set...")
    train_session_groups_list = []
    try:
        if all(level in train_segments_for_labels_df.index.names for level in grouping_cols):
            train_session_groups_list = list(train_segments_for_labels_df.groupby(level=grouping_cols))
        elif all(col in train_segments_for_labels_df.columns for col in grouping_cols): # Fallback if not in index
            train_session_groups_list = list(train_segments_for_labels_df.groupby(grouping_cols))
        else:
            raise ValueError(f"Could not find '{grouping_cols[0]}' and '{grouping_cols[1]}' for grouping in training data.")
        
        if verbose: print(f"   Total training session groups: {len(train_session_groups_list)}")
        if test_mode and train_session_groups_list:
            num_test_sessions = max(1, min(5, len(train_session_groups_list)))
            train_session_groups_list = train_session_groups_list[:num_test_sessions]
            print(f"   🧪 TEST MODE: Reduced to {len(train_session_groups_list)} training sessions.")
    except Exception as e_group:
        print(f"❌ Error during training set grouping: {e_group}. Check DataFrame structure.")
        sys.exit(1)

    print(f"\n🚀 Starting parallel training set processing ({len(train_session_groups_list)} groups)...")
    train_parallel_start_time = time.time()
    
    train_results_from_parallel = []
    if train_session_groups_list:
        try:
            train_results_from_parallel = Parallel(
                n_jobs=max_workers, verbose=(10 if verbose else 0), timeout=7200, prefer="processes"
            )(
                delayed(process_session_for_features)(
                    session_grp, DATA_ROOT / "train", SAMPLING_RATE,
                    num_channels_expected=expected_channels, verbose=verbose
                )
                for session_grp in tqdm(train_session_groups_list, desc="Train Sessions")
            )
            print(f"\n✅ Parallel training processing completed in {time.time() - train_parallel_start_time:.1f}s")
        except Exception as e_par_train:
            print(f"\n❌ ERROR in parallel training processing: {e_par_train}. Try reducing max_workers or check individual session processing.")
            train_results_from_parallel = []
    
    # Combine all training features and indices
    all_train_features_list: List[np.ndarray] = []
    all_train_original_indices: List[Any] = []
    
    if train_results_from_parallel:
        for result in train_results_from_parallel:
            if result and len(result) == 2:
                features_list, indices_list = result
                if features_list:
                    all_train_features_list.extend(features_list)
                if indices_list:
                    all_train_original_indices.extend(indices_list)

    # Convert to numpy arrays
    X_train_np = np.array([])
    y_train_np = np.array([])
    train_subject_ids_np = np.array([])
    
    if all_train_features_list:
        X_train_np = np.vstack(all_train_features_list)
        print(f"   ✅ Training features combined: {X_train_np.shape}")
    
    if all_train_original_indices:
        # Ensure indices are unique if there's any chance of duplicates from processing
        unique_train_indices = pd.Index(all_train_original_indices).unique()
        aligned_train_df = train_segments_for_labels_df.loc[unique_train_indices]
        y_train_np = aligned_train_df["label"].values
        if 'patient' in aligned_train_df.index.names:
            train_subject_ids_np = aligned_train_df.index.get_level_values('patient').to_numpy()
        elif 'patient' in aligned_train_df.columns:
            train_subject_ids_np = aligned_train_df['patient'].to_numpy()
    
    if verbose:
        print(f"\n📈 Training Set Results:")
        print(f"   X_train shape: {X_train_np.shape if X_train_np.size > 0 else 'Empty'}")
        print(f"   y_train shape: {y_train_np.shape if len(y_train_np) > 0 else 'Empty'}")
        print(f"   Train Subject IDs: {len(np.unique(train_subject_ids_np)) if len(train_subject_ids_np) > 0 else 'None'}")

    # --- Test Set Feature Extraction ---
    print("\n⏳ Processing Test Set...")
    test_session_groups_list = []
    try:
        if all(level in test_segments_df.index.names for level in grouping_cols):
            test_session_groups_list = list(test_segments_df.groupby(level=grouping_cols))
        elif all(col in test_segments_df.columns for col in grouping_cols):
            test_session_groups_list = list(test_segments_df.groupby(grouping_cols))
        else: raise ValueError(f"Could not find '{grouping_cols[0]}' and '{grouping_cols[1]}' for grouping in test data.")
        
        if verbose: print(f"   Total test session groups: {len(test_session_groups_list)}")
        if test_mode and test_session_groups_list:
            num_test_sessions = max(1, min(5, len(test_session_groups_list)))
            test_session_groups_list = test_session_groups_list[:num_test_sessions]
            print(f"   🧪 TEST MODE: Reduced to {len(test_session_groups_list)} test sessions.")
    except Exception as e_group_test:
        print(f"❌ Error during test set grouping: {e_group_test}. Check DataFrame structure.")
        sys.exit(1)

    print(f"\n🚀 Starting parallel test set processing ({len(test_session_groups_list)} groups)...")
    test_parallel_start_time = time.time()
    
    test_results_from_parallel = []
    if test_session_groups_list:
        try:
            test_results_from_parallel = Parallel(
                n_jobs=max_workers, verbose=(10 if verbose else 0), timeout=3600, prefer="processes"
            )(
                delayed(process_session_for_features)(
                    session_grp, DATA_ROOT / "test", SAMPLING_RATE,
                    num_channels_expected=expected_channels, verbose=verbose
                )
                for session_grp in tqdm(test_session_groups_list, desc="Test Sessions")
            )
            print(f"\n✅ Parallel test processing completed in {time.time() - test_parallel_start_time:.1f}s")
        except Exception as e_par_test:
            print(f"\n❌ ERROR in parallel test processing: {e_par_test}. Try reducing max_workers.")
            test_results_from_parallel = []

    # Combine all test features and indices
    all_test_features_list: List[np.ndarray] = []
    all_test_original_indices: List[Any] = []
    
    if test_results_from_parallel:
        for result in test_results_from_parallel:
            if result and len(result) == 2:
                features_list, indices_list = result
                if features_list:
                    all_test_features_list.extend(features_list)
                if indices_list:
                    all_test_original_indices.extend(indices_list)

    X_test_np = np.array([])
    if all_test_features_list:
        X_test_np = np.vstack(all_test_features_list)
        print(f"   ✅ Test features combined: {X_test_np.shape}")

    # Align X_test to the original order of test_segments_df, handling potential MultiIndex
    if all_test_original_indices and X_test_np.size > 0:
        temp_idx_for_df = None
        original_test_index = test_segments_df.index
        
        if isinstance(original_test_index, pd.MultiIndex):
            try:
                temp_idx_for_df = pd.MultiIndex.from_tuples(all_test_original_indices, names=original_test_index.names)
            except Exception as e_mi_create:
                if verbose: print(f"   Warning: Failed to create exact MultiIndex for temp_feature_df ({e_mi_create}). Fallback may occur.")
                # Fallback: try creating with default names if original names caused issue
                try:
                    num_levels = len(all_test_original_indices[0]) if all_test_original_indices and isinstance(all_test_original_indices[0], tuple) else 0
                    default_names = [f'level_{i}' for i in range(num_levels)] if num_levels > 0 else None
                    if default_names:
                        temp_idx_for_df = pd.MultiIndex.from_tuples(all_test_original_indices, names=default_names)
                except Exception as e_mi_fallback:
                     if verbose: print(f"   Warning: MultiIndex fallback creation also failed ({e_mi_fallback}). Alignment might be incorrect.")
        else: # Simple Index
            temp_idx_for_df = pd.Index(all_test_original_indices, name=original_test_index.name)

        if temp_idx_for_df is not None and X_test_np.ndim == 2: # Ensure X_test_np is 2D
            temp_feature_df = pd.DataFrame(X_test_np, index=temp_idx_for_df)
            # Reindex to match the original test_segments_df.index, filling missing with 0.0
            # This ensures X_test_np has one row for every row in test_segments_df, in the original order.
            try:
                aligned_X_test_df = temp_feature_df.reindex(original_test_index, fill_value=0.0)
                X_test_np = aligned_X_test_df.values
                if verbose: print(f"   X_test aligned to original test set. New shape: {X_test_np.shape}")
            except Exception as e_reindex:
                 if verbose: print(f"   Warning: Reindexing X_test failed ({e_reindex}). X_test might not be fully aligned.")
        elif X_test_np.ndim != 2 and X_test_np.size > 0:
             if verbose: print(f"   Warning: X_test_np is not 2D (shape: {X_test_np.shape}). Alignment skipped. Check feature vector consistency across sessions.")
        elif temp_idx_for_df is None and verbose:
             print("   Warning: Index for X_test alignment could not be created. X_test not aligned.")

    if verbose:
        print(f"\n🧪 Test Set Results:")
        print(f"   X_test shape: {X_test_np.shape if X_test_np.size > 0 else 'Empty'}")

    # --- Final Checks and Save ---
    print("\n--- Final Dataset Shapes: ---")
    print(f"X_train: {X_train_np.shape if X_train_np.size > 0 else 'Empty'}")
    print(f"y_train: {y_train_np.shape if len(y_train_np) > 0 else 'Empty'}")
    print(f"X_test: {X_test_np.shape if X_test_np.size > 0 else 'Empty'}")
    print(f"Train Subject IDs: {train_subject_ids_np.shape if len(train_subject_ids_np) > 0 else 'Empty'}")

    print("\n💾 Saving extracted feature arrays...")
    output_features_path = DATA_ROOT / "extracted_features"
    output_labels_path = DATA_ROOT / "labels"

    if X_test_np.size > 0: 
        np.save(output_features_path / "X_test.npy", X_test_np)
        print(f"   ✅ Saved X_test.npy: {X_test_np.shape}")
    else: 
        print("   X_test is empty, not saving.")
        
    if X_train_np.size > 0: 
        np.save(output_features_path / "X_train.npy", X_train_np)
        print(f"   ✅ Saved X_train.npy: {X_train_np.shape}")
    else: 
        print("   X_train is empty, not saving.")
        
    if len(y_train_np) > 0: 
        np.save(output_labels_path / "y_train.npy", np.asarray(y_train_np))
        print(f"   ✅ Saved y_train.npy: {y_train_np.shape}")
    else: 
        print("   y_train is empty, not saving.")
        
    if len(train_subject_ids_np) > 0: 
        np.save(output_features_path / "sample_subject_array_train.npy", train_subject_ids_np)
        print(f"   ✅ Saved train subject IDs: {train_subject_ids_np.shape}")
    else: 
        print("   train subject IDs is empty, not saving.")

    print(f"\n✅ Feature extraction and saving complete to {DATA_ROOT.resolve()}.")
    print(f"⏰ Total pipeline execution time: {time.time() - overall_pipeline_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Configure run parameters here:
    run_verbose = True  # Set to True for detailed logs
    run_test_mode = False # Set to True to process only a few sessions for quick testing
    run_max_workers = max(1, multiprocessing.cpu_count() - 2) # Leave 2 cores for system stability

    # Run the main feature extraction pipeline
    main(verbose=run_verbose, test_mode=run_test_mode, max_workers=run_max_workers if not run_test_mode else 2)
