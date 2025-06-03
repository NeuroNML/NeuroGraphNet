import os
import sys
import time
import multiprocessing
import psutil  # For memory monitoring
from pathlib import Path
from joblib import Parallel, delayed  # For parallel processing
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch, butter, iirnotch, sosfiltfilt, tf2sos
from scipy.stats import skew, kurtosis
# Removed scipy.fft import as fft, fftfreq are not directly used in the refactored feature extraction
# Replaced with scipy.stats.entropy for spectral entropy if your spectral_entropy utility is not standard
# Assuming src.utils.signal.spectral_entropy is similar to scipy.stats.entropy(psd_normalized)
# Keep if this is a specific, validated implementation
from src.utils.signal import spectral_entropy
# from scipy.stats import entropy as spectral_entropy # Alternative if src.utils.signal.spectral_entropy is not crucial

from src.utils.dataset import ensure_eeg_multiindex

# --- Core Feature Calculation ---


def _hjorth_parameters(signal: np.ndarray) -> Tuple[float, float]:
    """Calculate Hjorth mobility and complexity parameters."""
    if signal.size < 2:
        return 0.0, 0.0

    dx = np.diff(signal)
    var_x = np.var(signal)
    mobility = 0.0
    complexity = 0.0

    if var_x > 1e-10:  # Check variance to avoid division by zero or near-zero
        var_dx = np.var(dx)
        # Add epsilon for numerical stability
        mobility = np.sqrt(var_dx / (var_x + 1e-12))

        if var_dx > 1e-10 and dx.size >= 2:  # Need at least 2 points in dx for ddx
            ddx = np.diff(dx)
            # Ensure var_ddx is calculated only if ddx is not empty
            if ddx.size > 0:
                complexity = np.sqrt(np.var(ddx) / (var_dx + 1e-12))
            else:
                complexity = 0.0  # Or handle as appropriate if dx has only one element
    return mobility, complexity


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Calculate zero crossing rate."""
    if signal.size == 0:
        return 0.0
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def _sample_entropy(signal: np.ndarray, m: int = 2, r_coeff: float = 0.2) -> float:
    """
    Calculate sample entropy.
    This is a simplified version. For critical applications, consider a well-validated library.
    `r_coeff` is the coefficient for the tolerance r (e.g., 0.2 * std(signal)).
    """
    N = len(signal)
    if N < m + 1:
        return 0.0

    # Standard deviation of the original signal for tolerance r
    std_dev = np.std(signal)
    if std_dev < 1e-10:  # Signal is nearly flat
        return 0.0
    r = r_coeff * std_dev

    def _count_matches(m_len):
        # Create templates of length m_len
        templates = np.zeros((N - m_len + 1, m_len))
        for i in range(N - m_len + 1):
            templates[i, :] = signal[i: i + m_len]

        counts = 0
        # Iterate through unique pairs of templates
        for i in range(N - m_len + 1):
            # Max absolute difference for template i with all subsequent templates
            # Max over dimensions, then check if <= r, then sum
            # Taking abs difference for each element in vectors, then max of these differences
            max_abs_diff = np.max(
                np.abs(templates[i+1:, :] - templates[i, :]), axis=1)
            counts += np.sum(max_abs_diff <= r)
        return counts

    # Number of template pairs (m)
    # The factor of 2 in the original _phi seems to be double counting,
    # as (N-m+1)*(N-m)/2 is the number of unique pairs.
    # Here, _count_matches counts (N-m) comparisons for the first template, (N-m-1) for the second, etc.
    # sum_{i=0}^{N-m-1} (N-m-i) = (N-m)(N-m+1)/2
    # The definition of SampEn usually involves B_m / A_m where A_m is (N-m-1) choose 2 * C_m etc.
    # Or, more commonly, N_m / N_{m-1} (number of matching pairs)
    # Let's stick to a common definition:
    # C_m = number of pairs of vectors of length m such that d[x_i, x_j] <= r

    # Count for m
    count_m = _count_matches(m)
    # Count for m+1
    count_m1 = _count_matches(m + 1)

    if count_m == 0 or count_m1 == 0:
        return 0.0  # Or a very small number, or handle as per literature for such edge cases
    else:
        # The original formula _phi was count / (N - m + 1) / (N - m) * 2
        # Standard SampEn uses -log( (count_m1 / (N-m-1)(N-m)/2) / (count_m / (N-m)(N-m-1)/2) )
        # which simplifies to -log(count_m1 / count_m) if N is large
        # Or -log ( (num_pairs_m+1_match) / (num_pairs_m_match) )
        # A common simplification for SampEn is -log(B/A) where B is count_m1 and A is count_m
        # (assuming N-m is the number of template vectors for length m)
        # Let's use a common formulation: -log( (count_m1 / (N-m-1)) / (count_m / (N-m)) ) if N is large
        # For simplicity and following common approximate implementations:
        # Add epsilon to avoid log(0)
        return -np.log((count_m1 + 1e-10) / (count_m + 1e-10))


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, percentage: float = 0.95) -> float:
    """Calculate spectral edge frequency (frequency below which X% of power lies)."""
    if freqs.size == 0 or psd.size == 0 or freqs.size != psd.size:
        return 0.0
    total_power = np.sum(psd)
    if total_power < 1e-12:  # Check for zero or near-zero power
        return 0.0

    cumsum_power = np.cumsum(psd)
    threshold_power = total_power * percentage  # percentage is already 0.xx

    # Find the first index where cumulative power exceeds the threshold
    edge_indices = np.where(cumsum_power >= threshold_power)[0]
    if edge_indices.size > 0:
        return freqs[edge_indices[0]]
    # Should ideally not happen if percentage < 1.0 and total_power > 0
    return freqs[-1]


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Find the frequency with maximum power."""
    if freqs.size == 0 or psd.size == 0:
        return 0.0
    if len(psd) == 0:  # Should be caught by previous, but defensive
        return 0.0
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]

# --- Main Feature Extraction per Channel ---


# Define the number of features globally for consistency
# Time (12): mean, std, var, rms, p2p, skew, kurt, line_length, zcr, hj_mob, hj_cmp, samp_ent
# Freq (5 Abs B_Pow): delta, theta, alpha, beta, gamma
# Freq (5 Rel B_Pow): delta, theta, alpha, beta, gamma
# Freq (5 Spectral): spec_ent, peak_freq, spec_edge_50, spec_edge_95, total_power
# Freq (3 Ratios): alpha/beta, theta/alpha, delta/theta
# Total = 12 + 5 + 5 + 5 + 3 = 30 features
# The original script had 34 features.
# Removed: min_val, max_val (kept peak_to_peak), spectral_centroid, spectral_rolloff (which was spec_edge_95 again)
# Added: spec_edge_50 (median power frequency) as it's common.
# If spectral_centroid is preferred over spec_edge_50, it can be swapped.
# For this refactoring, I'll stick to 30 well-established features.
NUM_FEATURES_PER_CHANNEL_REF = 30


def _extract_channel_features(
    channel_signal: np.ndarray,
    fs: int,
    bands: Dict[str, Tuple[float, float]]
) -> List[float]:
    """
    Extracts a focused set of EEG features from a single channel,
    prioritizing those commonly used and effective for seizure detection.
    """
    if len(channel_signal) < fs / 2:  # Need at least half a second for meaningful spectral analysis
        return [0.0] * NUM_FEATURES_PER_CHANNEL_REF

    # === TIME DOMAIN FEATURES ===
    mean_val = np.mean(channel_signal)
    std_val = np.std(channel_signal)
    variance = np.var(channel_signal)  # or std_val**2
    rms_val = np.sqrt(np.mean(channel_signal**2))
    # same as np.max(channel_signal) - np.min(channel_signal)
    peak_to_peak = np.ptp(channel_signal)

    # Higher order statistics (require >1 point)
    skewness = skew(channel_signal) if len(channel_signal) > 1 else 0.0
    kurt_val = kurtosis(channel_signal) if len(
        channel_signal) > 1 else 0.0  # kurtosis, not kurt

    line_length = np.sum(np.abs(np.diff(channel_signal)))
    zero_cross_rate = _zero_crossing_rate(channel_signal)
    hj_mob, hj_cmp = _hjorth_parameters(channel_signal)
    sample_ent = _sample_entropy(channel_signal)  # Computationally intensive

    # === FREQUENCY DOMAIN FEATURES ===
    # Ensure nperseg is not greater than signal length and is reasonable
    # Max 2-second window for Welch, or signal length
    nperseg_welch = min(len(channel_signal), fs * 2)

    # Initialize spectral features
    abs_band_powers = {name: 0.0 for name in bands}
    rel_band_powers = {name: 0.0 for name in bands}
    spec_ent_val = 0.0
    peak_freq = 0.0
    spec_edge_50 = 0.0  # Median power frequency
    spec_edge_95 = 0.0  # 95% power frequency
    total_power_val = 0.0

    # Band ratios
    alpha_beta_ratio = 0.0
    theta_alpha_ratio = 0.0
    delta_theta_ratio = 0.0

    # Ensure signal is long enough for Welch
    if len(channel_signal) >= nperseg_welch and nperseg_welch > 0:
        try:
            freqs, psd = welch(channel_signal, fs=fs,
                               nperseg=nperseg_welch, scaling='density')

            if freqs.size > 0 and psd.size > 0:  # Check if welch returned valid output
                # More accurate total power from continuous PSD
                total_power_val = np.trapezoid(psd, freqs)
                if total_power_val < 1e-12:
                    total_power_val = 1e-12  # Avoid division by zero

                # Absolute and Relative Band powers
                for name, (lo, hi) in bands.items():
                    idx_band = (freqs >= lo) & (freqs < hi)
                    if np.any(idx_band):
                        abs_band_powers[name] = np.trapezoid(
                            psd[idx_band], freqs[idx_band])
                        rel_band_powers[name] = abs_band_powers[name] / \
                            total_power_val
                    else:
                        abs_band_powers[name] = 0.0
                        rel_band_powers[name] = 0.0

                # Spectral entropy (requires normalized PSD)
                # Normalize for entropy calculation
                psd_normalized = psd / (total_power_val + 1e-12)
                # Assuming spectral_entropy is from src.utils.signal or scipy.stats.entropy
                # Use only positive PSD values
                spec_ent_val = spectral_entropy(
                    psd_normalized[psd_normalized > 0])

                # Peak frequency
                peak_freq = _peak_frequency(freqs, psd)

                # Spectral edge frequencies
                spec_edge_50 = _spectral_edge_frequency(
                    freqs, psd, 0.50)  # Median power
                spec_edge_95 = _spectral_edge_frequency(freqs, psd, 0.95)

                # Band ratios
                if abs_band_powers.get("beta", 0.0) > 1e-12:
                    alpha_beta_ratio = abs_band_powers.get(
                        "alpha", 0.0) / abs_band_powers.get("beta", 0.0)
                if abs_band_powers.get("alpha", 0.0) > 1e-12:
                    theta_alpha_ratio = abs_band_powers.get(
                        "theta", 0.0) / abs_band_powers.get("alpha", 0.0)
                if abs_band_powers.get("theta", 0.0) > 1e-12:
                    delta_theta_ratio = abs_band_powers.get(
                        "delta", 0.0) / abs_band_powers.get("theta", 0.0)

        except (ValueError, FloatingPointError) as e:  # Catch specific errors
            # print(f"Warning: Welch PSD calculation failed for a channel: {e}") # Optional: log this
            pass  # Features will remain 0.0

    features = [
        # Time domain (12)
        mean_val, std_val, variance, rms_val, peak_to_peak,
        skewness, kurt_val, line_length, zero_cross_rate,
        hj_mob, hj_cmp, sample_ent,

        # Absolute band powers (5) - Ensure order matches `bands` typical order
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
    ]

    # Ensure the number of features matches NUM_FEATURES_PER_CHANNEL_REF
    if len(features) != NUM_FEATURES_PER_CHANNEL_REF:
        # This should not happen if logic is correct, but as a fallback:
        # print(f"Warning: Feature count mismatch. Expected {NUM_FEATURES_PER_CHANNEL_REF}, got {len(features)}")
        # Pad with zeros or truncate if necessary, though fixing the list is better
        padded_features = [0.0] * NUM_FEATURES_PER_CHANNEL_REF
        padded_features[:len(features)] = features
        return padded_features

    return features


def _extract_segment_features(segment_signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """Extract features from all channels in a segment using the refactored feature set."""
    all_channel_features: List[float] = []
    # Standard EEG bands
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (
        # Keep gamma up to 50Hz, consistent with BP filter
        13, 30), "gamma": (30, 50)}
    num_channels = segment_signal.shape[1]

    for ch_idx in range(num_channels):
        ch_signal_data = segment_signal[:, ch_idx]
        ch_feats = _extract_channel_features(ch_signal_data, fs, bands)
        all_channel_features.extend(ch_feats)

    features_array = np.asarray(all_channel_features, dtype=float)
    return np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)


def process_session_for_features(
    session_group: Tuple[Any, pd.DataFrame],
    base_signal_path: Path,
    bp_filter_coeffs: np.ndarray,
    notch_filter_coeffs: Optional[np.ndarray],
    f_s: int = 250,
    num_channels: int = 19,  # This should be derived or passed if variable
    # num_features_per_channel is now NUM_FEATURES_PER_CHANNEL_REF
    verbose: bool = False,
    debug: bool = True
) -> Tuple[List[np.ndarray], List[Any]]:
    group_name, session_df = session_group
    start_time = time.time()
    process_id = os.getpid()

    # Use the global refactored feature count
    num_features_per_channel_actual = NUM_FEATURES_PER_CHANNEL_REF

    status_msg = f"🔄 [PID:{process_id}] Starting job for session group: {group_name} at {time.strftime('%H:%M:%S')}"
    if debug:  # Simplified logging
        print(status_msg)
    elif verbose:
        print(status_msg)
        print(f"    📊 Session has {len(session_df)} segments to process")
        print(
            f"    🗂️ Expected features per segment: {num_channels * num_features_per_channel_actual}")

    last_status_time = time.time()

    def check_status(stage, force=False):
        nonlocal last_status_time
        current_time = time.time()
        elapsed = current_time - start_time

        if force or (current_time - last_status_time) > 30:
            mem = psutil.Process(
                os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            status = f"    🕒 [PID:{process_id}] {group_name} {stage} - Running for {elapsed:.1f}s - Mem: {mem:.1f}MB"
            print(status)
            last_status_time = current_time
            return True
        return False

    expected_feature_size = num_channels * num_features_per_channel_actual
    empty_features = np.zeros(expected_feature_size, dtype=float)

    features_list_for_session: List[np.ndarray] = []
    all_original_indices: List[Any] = []

    if 'signals_path' not in session_df.columns or session_df.empty:
        if verbose or debug:
            print(
                f"    ⚠️  No signals_path column or empty session for {group_name}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        elapsed_job = time.time() - start_time
        if verbose or debug:
            print(
                f"    ✅ {group_name} completed in {elapsed_job:.2f}s (empty session)")
        return features_list_for_session, all_original_indices

    relative_signal_file = session_df['signals_path'].iloc[0]
    full_session_signal_path = base_signal_path / relative_signal_file

    if verbose or debug:
        print(
            f"    📁 Loading signal file: {relative_signal_file} (Full: {full_session_signal_path})")

    if not full_session_signal_path.exists():
        if verbose or debug:
            print(f"    ❌ Signal file not found: {full_session_signal_path}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        elapsed_job = time.time() - start_time
        if verbose or debug:
            print(
                f"    ✅ {group_name} completed in {elapsed_job:.2f}s (file not found)")
        return features_list_for_session, all_original_indices

    try:
        load_start = time.time()
        if verbose or debug:
            print(f"    🔄 Reading parquet file...")
        check_status("Reading parquet")
        session_signal_df = pd.read_parquet(full_session_signal_path)
        session_signal_values = session_signal_df.values
        load_time = time.time() - load_start
        if verbose or debug:
            print(
                f"    ✅ Signal loaded: shape {session_signal_values.shape} in {load_time:.2f}s")
    except Exception as e:
        print(
            f"❌ Error loading {full_session_signal_path} for {group_name}: {e}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        elapsed_job = time.time() - start_time
        if verbose or debug:
            print(
                f"    ✅ {group_name} completed in {elapsed_job:.2f}s (load error)")
        return features_list_for_session, all_original_indices

    filter_start = time.time()
    if verbose or debug:
        print(f"    🔧 Applying filters...")

    check_status("Applying bandpass filter")
    processed_signal = sosfiltfilt(
        bp_filter_coeffs, session_signal_values, axis=0)
    if notch_filter_coeffs is not None:
        check_status("Applying notch filter")
        processed_signal = sosfiltfilt(
            notch_filter_coeffs, processed_signal, axis=0)

    check_status("Re-referencing and normalizing")
    avg_reference = np.mean(processed_signal, axis=1, keepdims=True)
    processed_signal = processed_signal - avg_reference
    mean_per_channel = np.mean(processed_signal, axis=0, keepdims=True)
    std_per_channel = np.std(processed_signal, axis=0, keepdims=True)
    processed_signal = (processed_signal - mean_per_channel) / \
        (std_per_channel + 1e-6)  # Add epsilon

    filter_time = time.time() - filter_start
    if verbose or debug:
        print(
            f"    ✅ Filtering completed in {filter_time:.2f}s. Processing {len(session_df)} segments...")

    segment_proc_start_time = time.time()
    segment_count = 0
    valid_segments = 0
    failed_segments = 0

    for original_idx, row in session_df.iterrows():
        segment_count += 1
        if verbose or segment_count % 50 == 0:  # Less frequent status for non-verbose
            check_status(
                f"Segment loop iter {segment_count}/{len(session_df)}", force=verbose and segment_count <= 5)

        try:
            t0 = int(row["start_time"] * f_s)
            tf = int(row["end_time"] * f_s)
            max_len = processed_signal.shape[0]

            if not (0 <= t0 < max_len and t0 < tf):
                if verbose and segment_count <= 3:
                    print(
                        f"          Segment {segment_count} Invalid time: t0={t0}, tf={tf}, max_len={max_len}")
                features_list_for_session.append(empty_features.copy())
                all_original_indices.append(original_idx)
                failed_segments += 1
                continue

            tf = min(tf, max_len)  # Ensure tf does not exceed max_len
            segment = processed_signal[t0:tf, :]

            # Require at least 0.1s of data for meaningful features
            if segment.shape[0] < f_s / 10:
                if verbose and segment_count <= 3:
                    print(
                        f"          Segment {segment_count} too short: {segment.shape[0]} samples")
                features_list_for_session.append(empty_features.copy())
                all_original_indices.append(original_idx)
                failed_segments += 1
                continue

            if verbose and segment_count <= 3:
                check_status(
                    f"Segment {segment_count} - Before _extract_segment_features", force=True)

            features = _extract_segment_features(segment, fs=f_s)

            if verbose and segment_count <= 3:
                check_status(
                    f"Segment {segment_count} - After _extract_segment_features", force=True)
                print(
                    f"          Features extracted: {len(features)} vals. Range: [{np.min(features):.2f}, {np.max(features):.2f}]")

            features_list_for_session.append(features)
            all_original_indices.append(original_idx)
            valid_segments += 1

        except Exception as e:
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
            failed_segments += 1
            if verbose or debug:
                print(
                    f"        ❌ Error processing segment {segment_count} (idx {original_idx}) for {group_name}: {e}")
            continue

    total_job_time = time.time() - start_time
    segment_processing_duration = time.time() - segment_proc_start_time

    if verbose or debug:
        print(
            f"    ✅ [PID:{process_id}] Session {group_name} completed in {total_job_time:.2f}s:")
        print(
            f"        Total segments: {segment_count}, Valid: {valid_segments}, Failed: {failed_segments} (Success: {valid_segments/segment_count*100 if segment_count > 0 else 0:.1f}%)")
        print(
            f"        Segment processing part took: {segment_processing_duration:.2f}s")
    elif segment_count > 0:  # Brief summary if not verbose
        print(
            f"    [PID:{process_id}] {group_name}: {valid_segments}/{segment_count} valid ({total_job_time:.2f}s)")

    return features_list_for_session, all_original_indices


IS_SCITAS = False  # Assuming this is set appropriately elsewhere
CPU_COUNT = multiprocessing.cpu_count()


def main(verbose: bool = False, test_mode: bool = False, max_workers: int = 4):
    print(f"\n🧠 EEG FEATURE EXTRACTION PIPELINE (Refactored)")
    print(f"{'='*60}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 System info: {CPU_COUNT} CPU cores available")

    # Use the refactored number of features
    current_features_per_channel = NUM_FEATURES_PER_CHANNEL_REF
    print(f"📈 Features per channel: {current_features_per_channel}")

    max_workers = min(max_workers, CPU_COUNT, 8)  # Cap workers, e.g. at 8
    print(
        f"🔧 Configuration: verbose={verbose}, test_mode={test_mode}, max_workers={max_workers}")

    LOCAL_DATA_ROOT = Path("data")
    (LOCAL_DATA_ROOT / "extracted_features").mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "labels").mkdir(parents=True, exist_ok=True)

    # DATA_ROOT logic remains the same
    # Or your actual SCITAS path
    DATA_ROOT = Path("/home/ogut/data") if IS_SCITAS else Path("data")
    # Assuming local processing for this example
    print(f"🔍 Looking for data in: {LOCAL_DATA_ROOT.resolve()}")

    if test_mode:
        print(f"\n🧪 TEST MODE ENABLED - Processing limited data")

    if verbose:
        print(f"\n📊 REFINED EEG FEATURE EXTRACTION")
        print(f"   Features per channel: {current_features_per_channel}")
        print(f"   Feature categories:")
        print(f"     • Time Domain: 12 features")
        print(f"     • Abs/Rel Band Powers: 5+5 features")
        print(f"     • Spectral Stats: 5 features (entropy, peak freq, edge freqs, total power)")
        print(f"     • Band Ratios: 3 features")
        print(f"   Parallel processing: {max_workers} workers")
        print(f"{'='*50}\n")

    try:
        clips_tr_full = pd.read_parquet(
            DATA_ROOT / "train" / "segments.parquet")
        clips_te_full = pd.read_parquet(
            DATA_ROOT / "test" / "segments.parquet")
        if verbose:
            print(
                f"✅ Successfully loaded dataset files: Train {clips_tr_full.shape}, Test {clips_te_full.shape}")
    except FileNotFoundError as e:
        print(f"❌ Error: Parquet file not found. Details: {e}")
        sys.exit(1)

    clips_tr_full = ensure_eeg_multiindex(clips_tr_full, id_col_name='id')
    clips_te_full = ensure_eeg_multiindex(clips_te_full, id_col_name='id')

    clips_tr_for_labels = clips_tr_full[~clips_tr_full.label.isna()].copy()
    clips_te = clips_te_full.copy()
    if verbose:
        print(
            f"   Training set after label filtering: {clips_tr_for_labels.shape}")
        print(f"   Test set: {clips_te.shape}")

    F_S = 250
    BP_FILTER_FREQS = (0.5, 50.0)  # Keep consistent with gamma band max
    NOTCH_FREQ_HZ = 60.0  # Or 50.0 depending on region
    NOTCH_Q_FACTOR = 30.0

    print(
        f"\n🔧 Designing filters: BP={BP_FILTER_FREQS}Hz, Notch={NOTCH_FREQ_HZ}Hz, FS={F_S}Hz")
    bp_filter_coeffs_sos = butter(
        4, BP_FILTER_FREQS, btype="bandpass", output="sos", fs=F_S)
    notch_filter_coeffs_ba = iirnotch(
        w0=NOTCH_FREQ_HZ, Q=NOTCH_Q_FACTOR, fs=F_S)
    notch_filter_coeffs_sos = tf2sos(*notch_filter_coeffs_ba)

    # --- Training Set Feature Extraction ---
    print("\n⏳ Processing Training Set...")
    if verbose:
        # Assuming 19 channels
        print(
            f"   Total features per segment: {19 * current_features_per_channel}")

    train_session_groups = []
    group_by_levels = ["patient", "session"]
    try:
        if all(level in clips_tr_for_labels.index.names for level in group_by_levels):
            train_session_groups = list(
                clips_tr_for_labels.groupby(level=group_by_levels))
        elif all(col in clips_tr_for_labels.columns for col in group_by_levels):
            train_session_groups = list(
                clips_tr_for_labels.groupby(group_by_levels))
        else:
            raise ValueError(
                f"Could not find {group_by_levels} for grouping in training data.")

        if verbose:
            print(
                f"   Total training session groups: {len(train_session_groups)}")
        if test_mode:
            train_session_groups = train_session_groups[:max(
                # Ensure at least 1 for testing
                1, min(5, len(train_session_groups)))]
            print(
                f"   🧪 TEST MODE: Reduced to {len(train_session_groups)} training sessions")
    except Exception as e:
        print(f"❌ Error during groupby for training set: {e}")
        sys.exit(1)

    print(
        f"\n🚀 Starting parallel training set processing with {max_workers} workers for {len(train_session_groups)} groups...")
    overall_start_time = time.time()

    train_processing_results = []
    try:
        train_processing_results = Parallel(
            n_jobs=max_workers,
            verbose=10 if verbose else 0,
            timeout=3600,  # 1 hour
            prefer="processes",
            batch_size=1
        )(
            delayed(process_session_for_features)(
                session_group,
                base_signal_path=LOCAL_DATA_ROOT / "train",  # Adjusted to LOCAL_DATA_ROOT
                bp_filter_coeffs=bp_filter_coeffs_sos,
                notch_filter_coeffs=notch_filter_coeffs_sos,
                f_s=F_S,
                # num_channels will be inferred by the data in process_session_for_features
                # num_features_per_channel is now globally defined
                verbose=verbose,
                debug=verbose  # Pass verbose as debug for more detailed logs from worker
            )
            for session_group in tqdm(train_session_groups, desc="Processing Train Sessions")
        )
        print(
            f"\n✅ Parallel training processing completed successfully in {time.time() - overall_start_time:.1f}s")
    except Exception as e:
        print(
            f"\n❌ ERROR in parallel training processing: {e}. Failed after {time.time() - overall_start_time:.1f}s")
        # Fallback logic can be reinstated here if needed

    X_train_list = []
    all_train_indices = []
    for res_list, idx_list in train_processing_results:
        X_train_list.extend(res_list)
        all_train_indices.extend(idx_list)

    X_train = np.array(X_train_list) if X_train_list else np.array([])
    y_train = np.array([])
    sample_subject_list_train = np.array([])

    if all_train_indices:
        clips_tr_aligned = clips_tr_for_labels.loc[all_train_indices]
        y_train = clips_tr_aligned["label"].values
        # Extract subject list (assuming 'patient' is the subject identifier)
        if 'patient' in clips_tr_aligned.index.names:
            sample_subject_list_train = clips_tr_aligned.index.get_level_values(
                'patient').to_numpy()
        elif 'patient' in clips_tr_aligned.columns:
            sample_subject_list_train = clips_tr_aligned['patient'].to_numpy()

    if verbose:
        print(f"\n📈 Training set feature extraction results:")
        print(
            f"   Features (X_train): {X_train.shape if X_train.size > 0 else 'None'}")
        print(
            f"   Labels (y_train): {y_train.shape if y_train.size > 0 else 'None'}")
        print(
            f"   Subjects: {len(np.unique(sample_subject_list_train)) if sample_subject_list_train.size > 0 else 'None'}")

    # --- Test Set Feature Extraction (similar structure) ---
    print("\n⏳ Processing Test Set...")
    test_session_groups = []
    try:
        if all(level in clips_te.index.names for level in group_by_levels):
            test_session_groups = list(clips_te.groupby(level=group_by_levels))
        elif all(col in clips_te.columns for col in group_by_levels):
            test_session_groups = list(clips_te.groupby(group_by_levels))
        else:
            raise ValueError(
                f"Could not find {group_by_levels} for grouping in test data.")

        if verbose:
            print(f"   Total test session groups: {len(test_session_groups)}")
        # Apply test mode to test set as well
        if test_mode and len(test_session_groups) > 5:
            original_test_count = len(test_session_groups)
            test_session_groups = test_session_groups[:max(
                1, min(5, len(test_session_groups)))]
            print(
                f"   🧪 TEST MODE: Reduced from {original_test_count} to {len(test_session_groups)} test sessions")

    except Exception as e:
        print(f"❌ Error during groupby for test set: {e}")
        sys.exit(1)

    print(
        f"\n🚀 Starting parallel test set processing with {max_workers} workers for {len(test_session_groups)} groups...")
    overall_start_time_test = time.time()

    test_processing_results = []
    try:
        test_processing_results = Parallel(
            n_jobs=max_workers,
            verbose=10 if verbose else 0,
            timeout=1800,  # 30 mins
            prefer="processes",
            batch_size=1
        )(
            delayed(process_session_for_features)(
                session_group,
                base_signal_path=LOCAL_DATA_ROOT / "test",  # Adjusted to LOCAL_DATA_ROOT
                bp_filter_coeffs=bp_filter_coeffs_sos,
                notch_filter_coeffs=notch_filter_coeffs_sos,
                f_s=F_S,
                verbose=verbose,
                debug=verbose
            )
            for session_group in tqdm(test_session_groups, desc="Processing Test Sessions")
        )
        print(
            f"\n✅ Parallel test processing completed successfully in {time.time() - overall_start_time_test:.1f}s")
    except Exception as e:
        print(
            f"\n❌ ERROR in parallel test processing: {e}. Failed after {time.time() - overall_start_time_test:.1f}s")
        # Fallback logic can be reinstated here if needed

    X_test_list = []
    # For test set, we might need all original indices to align if submitting predictions
    all_test_indices = []
    for res_list, idx_list in test_processing_results:
        X_test_list.extend(res_list)
        all_test_indices.extend(idx_list)  # Collect all original indices

    X_test = np.array(X_test_list) if X_test_list else np.array([])

    # If you need to align X_test with the original clips_te order and fill gaps:
    if all_test_indices and X_test.size > 0:
        # Create a DataFrame with extracted features and their original indices
        # Adjust if num_channels is dynamic
        num_total_features = 19 * current_features_per_channel
        temp_feature_df = pd.DataFrame(X_test, index=pd.Index(
            all_test_indices, name=clips_te.index.name or 'original_index'))
        # Reindex to match the original clips_te DataFrame, filling missing rows with zeros (or NaNs)
        # This assumes clips_te.index is unique and suitable for .loc
        # Ensure the index names match if they are multi-indices.
        # For simplicity, if clips_te.index is a simple index:
        if isinstance(clips_te.index, pd.MultiIndex):
            print(
                "Warning: X_test alignment for MultiIndex needs careful handling of index levels.")
            # Potentially convert multi-index to a single temporary column for merging/joining if needed
        else:  # Assuming simple index
            aligned_X_test_df = temp_feature_df.reindex(
                clips_te.index)  # Fill with NaN by default
            # Fill NaNs that arose from reindexing (segments that failed processing or were not in results)
            aligned_X_test_df = aligned_X_test_df.fillna(
                0.0)  # Or appropriate fill value
            X_test = aligned_X_test_df.values
            print(
                f"   X_test aligned to original test set shape: {X_test.shape}")

    if verbose:
        print(f"\n🧪 Test set feature extraction results:")
        print(
            f"   Features (X_test): {X_test.shape if X_test.size > 0 else 'None'}")

    # --- Final Checks and Save ---
    print("\n--- Final dataset shapes: ---")
    print(f"X_train: {X_train.shape if X_train.size > 0 else 'Empty'}")
    print(f"y_train: {y_train.shape if y_train.size > 0 else 'Empty'}")
    print(f"X_test: {X_test.shape if X_test.size > 0 else 'Empty'}")
    print(
        f"sample_subject_list_train: {sample_subject_list_train.shape if sample_subject_list_train.size > 0 else 'Empty'}")

    print("\n💾 Saving extracted feature arrays...")
    if X_test.size > 0:
        np.save(LOCAL_DATA_ROOT / "extracted_features/X_test.npy", X_test)
    else:
        print("X_test is empty, not saving.")
    if X_train.size > 0:
        np.save(LOCAL_DATA_ROOT / "extracted_features/X_train.npy", X_train)
    else:
        print("X_train is empty, not saving.")
    if y_train.size > 0:
        np.save(LOCAL_DATA_ROOT / "labels/y_train.npy", y_train)
    else:
        print("y_train is empty, not saving.")
    if sample_subject_list_train.size > 0:
        np.save(LOCAL_DATA_ROOT / "extracted_features/sample_subject_array_train.npy",
                sample_subject_list_train)
    else:
        print("sample_subject_list_train is empty, not saving.")

    print("✅ Feature extraction and saving complete.")
    print(
        f"⏰ Total pipeline execution time: {time.time() - overall_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Example: Run with verbose and test_mode enabled
    main(verbose=False, test_mode=False, max_workers=multiprocessing.cpu_count() -1 )
