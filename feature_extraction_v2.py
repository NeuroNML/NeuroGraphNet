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
# Assuming src.utils.signal.spectral_entropy is a custom, validated implementation
# If not, you might consider from scipy.stats import entropy as spectral_entropy
# and ensure psd is normalized before passing.
from src.utils.signal import spectral_entropy
from src.utils.dataset import ensure_eeg_multiindex

# --- Configuration & Global Constants ---
# Define the number of features globally for consistency
# Time (12): mean, std, var, rms, p2p, skew, kurt, line_length, zcr, hj_mob, hj_cmp, samp_ent
# Freq (5 Abs B_Pow): delta, theta, alpha, beta, gamma
# Freq (5 Rel B_Pow): delta, theta, alpha, beta, gamma
# Freq (5 Spectral): spec_ent, peak_freq, spec_edge_50, spec_edge_95, total_power
# Freq (3 Ratios): alpha/beta, theta/alpha, delta/theta
# Total = 12 + 5 + 5 + 5 + 3 = 30 features
NUM_FEATURES_PER_CHANNEL_REF = 30
DEFAULT_EXPECTED_CHANNELS = 19  # Default, can be overridden if known

# --- Core Feature Calculation Helper Functions ---


def _hjorth_parameters(signal: np.ndarray) -> Tuple[float, float]:
    """Calculate Hjorth mobility and complexity parameters."""
    if signal.size < 2:  # Need at least 2 points for diff
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
            # Ensure ddx is not empty (dx had at least 2 points)
            if ddx.size > 0:
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
    if N <= m:  # Need N > m for templates of length m+1
        return 0.0

    std_dev = np.std(signal)
    if std_dev < 1e-10:  # Signal is nearly flat, entropy is low or undefined
        return 0.0
    r_tolerance = r_coeff * std_dev

    def _count_matches(current_m_len: int) -> int:
        """Counts pairs of templates of length `current_m_len` that are similar."""
        if N <= current_m_len:
            return 0

        # Create templates (subsequences) of length current_m_len
        templates = np.zeros((N - current_m_len + 1, current_m_len))
        for i in range(N - current_m_len + 1):
            templates[i, :] = signal[i: i + current_m_len]

        num_templates = templates.shape[0]
        counts = 0
        # Iterate through unique pairs of templates (i < j)
        for i in range(num_templates):
            for j in range(i + 1, num_templates):
                # Calculate Chebyshev distance (max absolute difference)
                max_abs_diff = np.max(
                    np.abs(templates[i, :] - templates[j, :]))
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
        return 0.0  # Invalid input

    # Ensure psd values are non-negative
    psd_sanitized = np.maximum(psd, 0)
    # Integrate using trapezoidal rule
    total_power = np.trapz(psd_sanitized, freqs)

    if total_power < 1e-12:  # Check for zero or near-zero power
        return 0.0  # Or freqs[0] if that's more appropriate

    # Approximate integration for cumsum
    cumulative_power = np.cumsum(psd_sanitized) * \
        (freqs[1] - freqs[0] if len(freqs) > 1 else 1)
    # A more accurate way for cumulative power using trapz if freqs are not evenly spaced:
    if len(freqs) > 1:
        cumulative_power_values = [0.0] * len(freqs)
        for i in range(1, len(freqs)):
            cumulative_power_values[i] = np.trapz(
                psd_sanitized[:i+1], freqs[:i+1])
        cumulative_power = np.array(cumulative_power_values)
    else:  # Single frequency point
        cumulative_power = psd_sanitized * (freqs[0] if freqs.size > 0 else 1)

    threshold_abs_power = total_power * percentage_power

    # Find the first index where cumulative power exceeds or equals the threshold
    edge_indices = np.where(cumulative_power >= threshold_abs_power)[0]

    if edge_indices.size > 0:
        return freqs[edge_indices[0]]

    # If threshold is not met (e.g., percentage_power = 1.0 and numerical precision issues)
    # or if all power is concentrated at the very end.
    return freqs[-1]


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Find the frequency with maximum power."""
    if freqs.size == 0 or psd.size == 0 or freqs.size != psd.size:
        return 0.0
    if psd.size == 0:  # Should be caught by previous, but defensive
        return 0.0
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]

# --- Main Feature Extraction per Channel ---


def _extract_channel_features(
    channel_signal: np.ndarray,
    fs: int,
    bands: Dict[str, Tuple[float, float]]
) -> List[float]:
    """
    Extracts a focused set of EEG features from a single channel,
    prioritizing those commonly used and effective for seizure detection.
    """
    # Need enough data for meaningful analysis, e.g., at least one full cycle of lowest band or for Welch window
    # e.g. 0.5 sec for Welch
    min_len_for_analysis = max(fs / bands.get("delta", (0.5, 4))[0], fs * 0.5)
    if len(channel_signal) < min_len_for_analysis:
        return [0.0] * NUM_FEATURES_PER_CHANNEL_REF

    # === TIME DOMAIN FEATURES ===
    mean_val = np.mean(channel_signal)
    std_val = np.std(channel_signal)
    variance = std_val**2  # More direct than np.var if std_val is already computed
    rms_val = np.sqrt(np.mean(channel_signal**2))
    # np.max(channel_signal) - np.min(channel_signal)
    peak_to_peak = np.ptp(channel_signal)

    skewness = skew(channel_signal) if len(channel_signal) > 1 else 0.0
    kurt_val = kurtosis(channel_signal, fisher=True) if len(
        channel_signal) > 1 else 0.0  # Fisher's definition (normal ==> 0)

    line_length = np.sum(np.abs(np.diff(channel_signal))
                         ) if len(channel_signal) > 1 else 0.0
    zero_cross_rate = _zero_crossing_rate(channel_signal)
    hj_mob, hj_cmp = _hjorth_parameters(channel_signal)
    # Can be computationally intensive
    sample_ent = _sample_entropy(channel_signal)

    # === FREQUENCY DOMAIN FEATURES ===
    # Max 2-second window for Welch, or signal length if shorter
    nperseg_welch = min(len(channel_signal), fs * 2)

    abs_band_powers = {name: 0.0 for name in bands}
    rel_band_powers = {name: 0.0 for name in bands}
    spec_ent_val = 0.0
    peak_freq = 0.0
    spec_edge_50 = 0.0  # Median power frequency
    spec_edge_95 = 0.0  # 95% power frequency
    total_power_val = 0.0
    alpha_beta_ratio, theta_alpha_ratio, delta_theta_ratio = 0.0, 0.0, 0.0

    if len(channel_signal) >= nperseg_welch and nperseg_welch > 0:
        try:
            freqs, psd = welch(
                channel_signal, fs=fs, nperseg=nperseg_welch, scaling='density', average='mean')

            if freqs.size > 0 and psd.size > 0:
                psd = np.maximum(psd, 0)  # Ensure non-negative PSD
                total_power_val = np.trapz(psd, freqs)
                if total_power_val < 1e-12:
                    total_power_val = 1e-12  # Avoid division by zero later

                for name, (lo, hi) in bands.items():
                    idx_band = (freqs >= lo) & (freqs < hi)
                    if np.any(idx_band):
                        abs_band_powers[name] = np.trapz(
                            psd[idx_band], freqs[idx_band])
                        rel_band_powers[name] = abs_band_powers[name] / \
                            total_power_val

                psd_normalized = psd / total_power_val  # Normalize for entropy
                # Use only positive, non-zero parts of PSD for entropy calculation
                psd_nz = psd_normalized[psd_normalized > 1e-12]
                if psd_nz.size > 0:
                    # Assuming spectral_entropy is from src.utils.signal or scipy.stats.entropy
                    # If using scipy.stats.entropy, it expects probabilities (sum to 1)
                    # Normalize by log2(len(psd_nz)) if needed
                    spec_ent_val = spectral_entropy(psd_nz, base=2)
                    # Or a custom implementation: -np.sum(psd_nz * np.log2(psd_nz))

                peak_freq = _peak_frequency(freqs, psd)
                spec_edge_50 = _spectral_edge_frequency(freqs, psd, 0.50)
                spec_edge_95 = _spectral_edge_frequency(freqs, psd, 0.95)

                if abs_band_powers.get("beta", 0.0) > 1e-12:
                    alpha_beta_ratio = abs_band_powers.get(
                        "alpha", 0.0) / abs_band_powers.get("beta", 0.0)
                if abs_band_powers.get("alpha", 0.0) > 1e-12:
                    theta_alpha_ratio = abs_band_powers.get(
                        "theta", 0.0) / abs_band_powers.get("alpha", 0.0)
                if abs_band_powers.get("theta", 0.0) > 1e-12:
                    delta_theta_ratio = abs_band_powers.get(
                        "delta", 0.0) / abs_band_powers.get("theta", 0.0)
        except (ValueError, FloatingPointError, IndexError) as e:
            # print(f"Debug: Welch PSD calculation or subsequent freq feature failed: {e}")
            pass  # Features will remain 0.0

    features = [
        mean_val, std_val, variance, rms_val, peak_to_peak,
        skewness, kurt_val, line_length, zero_cross_rate,
        hj_mob, hj_cmp, sample_ent,
        abs_band_powers.get("delta", 0.0), abs_band_powers.get("theta", 0.0),
        abs_band_powers.get("alpha", 0.0), abs_band_powers.get("beta", 0.0),
        abs_band_powers.get("gamma", 0.0),
        rel_band_powers.get("delta", 0.0), rel_band_powers.get("theta", 0.0),
        rel_band_powers.get("alpha", 0.0), rel_band_powers.get("beta", 0.0),
        rel_band_powers.get("gamma", 0.0),
        spec_ent_val, peak_freq, spec_edge_50, spec_edge_95, total_power_val,
        alpha_beta_ratio, theta_alpha_ratio, delta_theta_ratio,
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
    """Extract features from all channels in a segment using the refactored feature set."""
    all_channel_features: List[float] = []
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (
        8, 13), "beta": (13, 30), "gamma": (30, 50)}

    actual_num_channels = segment_signal.shape[1]
    if actual_num_channels == 0:  # Should not happen if segment is valid
        return np.array([0.0] * (DEFAULT_EXPECTED_CHANNELS * NUM_FEATURES_PER_CHANNEL_REF))

    for ch_idx in range(actual_num_channels):
        ch_signal_data = segment_signal[:, ch_idx]
        ch_feats = _extract_channel_features(ch_signal_data, fs, bands)
        all_channel_features.extend(ch_feats)

    features_array = np.asarray(all_channel_features, dtype=float)

    # If the number of actual channels differs from a fixed expectation,
    # the total length of features_array will vary. This is handled later during DataFrame creation.
    # For now, just ensure NaNs/Infs are handled.
    return np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)


def process_session_for_features(
    session_group: Tuple[Any, pd.DataFrame],
    base_signal_path: Path,
    bp_filter_coeffs: np.ndarray,
    notch_filter_coeffs: Optional[np.ndarray],
    f_s: int = 250,
    # Used for empty_features if signal load fails
    num_channels_expected: int = DEFAULT_EXPECTED_CHANNELS,
    verbose: bool = False,
    debug: bool = False  # Renamed from original debug for clarity
) -> Tuple[List[np.ndarray], List[Any]]:
    group_name, session_df = session_group
    job_start_time = time.time()
    process_id = os.getpid()

    num_features_per_channel = NUM_FEATURES_PER_CHANNEL_REF

    status_msg = f"🔄 [PID:{process_id}] Starting: {group_name} ({len(session_df)} segs) at {time.strftime('%H:%M:%S')}"
    if debug or verbose:
        print(status_msg)

    last_status_update_time = time.time()

    def _log_status_update(stage_msg, force_print=False):
        nonlocal last_status_update_time
        current_time = time.time()
        # Update every 30s or if forced
        if force_print or (current_time - last_status_update_time) > 30:
            mem_usage_mb = psutil.Process(
                os.getpid()).memory_info().rss / (1024 * 1024)
            elapsed_total = current_time - job_start_time
            print(
                f"    🕒 [PID:{process_id}] {group_name} - {stage_msg} - Elap: {elapsed_total:.1f}s - Mem: {mem_usage_mb:.1f}MB")
            last_status_update_time = current_time

    # Pre-allocate empty features based on *expected* channels.
    # If actual channels differ, _extract_segment_features will return a different length.
    empty_features_alloc = np.zeros(
        num_channels_expected * num_features_per_channel, dtype=float)

    features_list_for_session: List[np.ndarray] = []
    all_original_indices_for_session: List[Any] = []

    if 'signals_path' not in session_df.columns or session_df.empty:
        if verbose or debug:
            print(
                f"    ⚠️ No 'signals_path' or empty session for {group_name}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features_alloc.copy())
            all_original_indices_for_session.append(original_idx)
        if verbose or debug:
            print(
                f"    ✅ {group_name} done (empty session) in {time.time() - job_start_time:.2f}s")
        return features_list_for_session, all_original_indices_for_session

    relative_signal_file = session_df['signals_path'].iloc[0]
    full_signal_path = base_signal_path / relative_signal_file

    if verbose or debug:
        print(f"    📁 Loading: {full_signal_path}")

    if not full_signal_path.exists():
        if verbose or debug:
            print(f"    ❌ File not found: {full_signal_path}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features_alloc.copy())
            all_original_indices_for_session.append(original_idx)
        if verbose or debug:
            print(
                f"    ✅ {group_name} done (file not found) in {time.time() - job_start_time:.2f}s")
        return features_list_for_session, all_original_indices_for_session

    try:
        _log_status_update("Reading parquet", force_print=verbose)
        signal_load_start = time.time()
        session_eeg_data_df = pd.read_parquet(full_signal_path)
        # Shape: (time_points, actual_num_channels)
        session_eeg_values = session_eeg_data_df.values
        actual_num_channels_in_file = session_eeg_values.shape[1]
        if verbose or debug:
            print(
                f"    ✅ Loaded signal {session_eeg_values.shape} in {time.time() - signal_load_start:.2f}s")
    except Exception as e:
        print(f"    ❌ Error loading {full_signal_path} for {group_name}: {e}")
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(
                empty_features_alloc.copy())  # Use pre-allocated
            all_original_indices_for_session.append(original_idx)
        if verbose or debug:
            print(
                f"    ✅ {group_name} done (load error) in {time.time() - job_start_time:.2f}s")
        return features_list_for_session, all_original_indices_for_session

    # If actual channels differ from expected, create a specific empty_features for this session
    current_session_empty_features = empty_features_alloc
    if actual_num_channels_in_file != num_channels_expected:
        if verbose or debug:
            print(
                f"    ℹ️ Channel mismatch for {group_name}: Expected {num_channels_expected}, Got {actual_num_channels_in_file}. Adjusting empty features.")
        current_session_empty_features = np.zeros(
            actual_num_channels_in_file * num_features_per_channel, dtype=float)

    _log_status_update("Filtering", force_print=verbose)
    filter_proc_start = time.time()
    processed_signal_data = sosfiltfilt(
        bp_filter_coeffs, session_eeg_values, axis=0)
    if notch_filter_coeffs is not None:
        processed_signal_data = sosfiltfilt(
            notch_filter_coeffs, processed_signal_data, axis=0)

    avg_ref = np.mean(processed_signal_data, axis=1, keepdims=True)
    processed_signal_data = processed_signal_data - avg_ref
    mean_ch = np.mean(processed_signal_data, axis=0, keepdims=True)
    std_ch = np.std(processed_signal_data, axis=0, keepdims=True)
    processed_signal_data = (processed_signal_data - mean_ch) / \
        (std_ch + 1e-7)  # Epsilon for stability
    if verbose or debug:
        print(
            f"    ✅ Filtering & Normalization done in {time.time() - filter_proc_start:.2f}s")

    segment_loop_start_time = time.time()
    processed_segment_count = 0
    valid_segment_features_count = 0
    failed_segment_extraction_count = 0

    for original_idx, segment_meta_row in session_df.iterrows():
        processed_segment_count += 1
        if verbose and (processed_segment_count <= 3 or processed_segment_count % 50 == 0):
            _log_status_update(
                f"Segment {processed_segment_count}/{len(session_df)}", force_print=True)

        try:
            start_sample = int(segment_meta_row["start_time"] * f_s)
            end_sample = int(segment_meta_row["end_time"] * f_s)
            signal_max_len = processed_signal_data.shape[0]

            is_valid_segment_time = (
                0 <= start_sample < signal_max_len and start_sample < end_sample)
            if not is_valid_segment_time:
                if verbose and processed_segment_count <= 3:
                    print(
                        f"        Segment {processed_segment_count} Invalid time: start={start_sample}, end={end_sample}, max_len={signal_max_len}")
                features_list_for_session.append(
                    current_session_empty_features.copy())
                all_original_indices_for_session.append(original_idx)
                failed_segment_extraction_count += 1
                continue

            # Clip end_sample to signal length
            end_sample = min(end_sample, signal_max_len)
            eeg_segment_data = processed_signal_data[start_sample:end_sample, :]

            # Min length for feature extraction (e.g., 0.1 seconds)
            if eeg_segment_data.shape[0] < max(10, f_s // 10):
                if verbose and processed_segment_count <= 3:
                    print(
                        f"        Segment {processed_segment_count} too short: {eeg_segment_data.shape[0]} samples")
                features_list_for_session.append(
                    current_session_empty_features.copy())
                all_original_indices_for_session.append(original_idx)
                failed_segment_extraction_count += 1
                continue

            segment_features = _extract_segment_features(
                eeg_segment_data, fs=f_s)
            features_list_for_session.append(segment_features)
            all_original_indices_for_session.append(original_idx)
            valid_segment_features_count += 1

        except Exception as e_seg:
            features_list_for_session.append(
                current_session_empty_features.copy())
            all_original_indices_for_session.append(original_idx)
            failed_segment_extraction_count += 1
            if verbose or debug:
                print(
                    f"        ❌ Error processing segment {processed_segment_count} (orig_idx {original_idx}) for {group_name}: {e_seg}")
            continue

    total_job_duration = time.time() - job_start_time
    if verbose or debug:
        print(
            f"    ✅ [PID:{process_id}] {group_name} finished in {total_job_duration:.2f}s.")
        print(
            f"        Segments: Total={processed_segment_count}, ValidFeats={valid_segment_features_count}, FailedExtr={failed_segment_extraction_count}")
    elif processed_segment_count > 0:  # Brief summary if not verbose
        print(f"    [PID:{process_id}] {group_name}: {valid_segment_features_count}/{processed_segment_count} valid feats ({total_job_duration:.2f}s)")

    return features_list_for_session, all_original_indices_for_session


# --- Main Execution Logic ---
IS_SCITAS = False  # User specific: Set to True if running on SCITAS environment
CPU_COUNT = multiprocessing.cpu_count()


def main(verbose: bool = False, test_mode: bool = False, max_workers: Optional[int] = None):
    overall_pipeline_start_time = time.time()
    print(f"\n🧠 EEG FEATURE EXTRACTION PIPELINE (Full Refactored Script)")
    print(f"{'='*60}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 System info: {CPU_COUNT} CPU cores available")

    features_per_channel = NUM_FEATURES_PER_CHANNEL_REF
    # For printouts and default empty features
    expected_channels = DEFAULT_EXPECTED_CHANNELS
    print(
        f"📈 Features per channel: {features_per_channel}, Expected channels: {expected_channels}")

    if max_workers is None:
        max_workers = max(1, CPU_COUNT - 1 if CPU_COUNT > 1 else 1)
    # Cap workers, e.g., at 16 or a reasonable number
    max_workers = min(max_workers, CPU_COUNT, 16)
    print(
        f"🔧 Config: verbose={verbose}, test_mode={test_mode}, max_workers={max_workers}")

    # Define data paths
    # Adjust DATA_ROOT if not running from the project root or if data is elsewhere
    project_root = Path(__file__).resolve(
    ).parent.parent if "__file__" in locals() else Path.cwd()
    LOCAL_DATA_ROOT = project_root / "data"

    # SCITAS path needs to be configured by the user if IS_SCITAS is True
    # Placeholder: User should configure this
    SCITAS_DATA_ROOT = Path("/home/user/data")
    DATA_ROOT = SCITAS_DATA_ROOT if IS_SCITAS else LOCAL_DATA_ROOT

    print(f"🔍 Using DATA_ROOT: {DATA_ROOT.resolve()}")
    (DATA_ROOT / "extracted_features").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "labels").mkdir(parents=True, exist_ok=True)

    if test_mode:
        print(f"\n🧪 TEST MODE ENABLED - Processing limited data samples.")
    if verbose:
        print(f"\n📊 REFINED EEG FEATURE EXTRACTION DETAILS:")
        print(f"   Features per channel: {features_per_channel}")
        print(
            f"   Total expected features per segment (for {expected_channels} channels): {expected_channels * features_per_channel}")
        print(f"   Parallel processing with up to {max_workers} workers.")
        print(f"{'='*50}\n")

    try:
        clips_tr_full_df = pd.read_parquet(
            DATA_ROOT / "train" / "segments.parquet")
        clips_te_full_df = pd.read_parquet(
            DATA_ROOT / "test" / "segments.parquet")
        if verbose:
            print(
                f"✅ Loaded segment metadata: Train {clips_tr_full_df.shape}, Test {clips_te_full_df.shape}")
    except FileNotFoundError as e:
        print(
            f"❌ CRITICAL Error: Segments.parquet file not found. Path: {e.filename}. Ensure data is in {DATA_ROOT}/train and {DATA_ROOT}/test.")
        sys.exit(1)

    clips_tr_full_df = ensure_eeg_multiindex(
        clips_tr_full_df, id_col_name='id')
    clips_te_full_df = ensure_eeg_multiindex(
        clips_te_full_df, id_col_name='id')

    train_segments_for_labels_df = clips_tr_full_df[~clips_tr_full_df['label'].isna(
    )].copy()
    test_segments_df = clips_te_full_df.copy()
    if verbose:
        print(
            f"   Training segments (valid labels): {train_segments_for_labels_df.shape}")
        print(f"   Test segments: {test_segments_df.shape}")

    SAMPLING_RATE = 250
    BANDPASS_FREQS = (0.5, 50.0)  # Consistent with Gamma band upper limit
    NOTCH_FILTER_HZ = 60.0  # Or 50.0 depending on region of data origin
    NOTCH_Q = 30.0

    print(
        f"\n🔧 Designing filters: BP={BANDPASS_FREQS}Hz, Notch={NOTCH_FILTER_HZ}Hz (Q={NOTCH_Q}), FS={SAMPLING_RATE}Hz")
    bp_sos_coeffs = butter(
        4, BANDPASS_FREQS, btype="bandpass", output="sos", fs=SAMPLING_RATE)
    notch_b_a_coeffs = iirnotch(
        w0=NOTCH_FILTER_HZ, Q=NOTCH_Q, fs=SAMPLING_RATE)
    notch_sos_coeffs = tf2sos(*notch_b_a_coeffs)

    # For grouping segments by session signal file
    grouping_cols = ["patient", "session"]

    # --- Training Set Feature Extraction ---
    print("\n⏳ Processing Training Set...")
    train_session_groups_list = []
    try:
        if all(level in train_segments_for_labels_df.index.names for level in grouping_cols):
            train_session_groups_list = list(
                train_segments_for_labels_df.groupby(level=grouping_cols))
        # Fallback if not in index
        elif all(col in train_segments_for_labels_df.columns for col in grouping_cols):
            train_session_groups_list = list(
                train_segments_for_labels_df.groupby(grouping_cols))
        else:
            raise ValueError(
                f"Could not find '{grouping_cols[0]}' and '{grouping_cols[1]}' for grouping in training data.")

        if verbose:
            print(
                f"   Total training session groups: {len(train_session_groups_list)}")
        if test_mode and train_session_groups_list:
            num_test_sessions = max(1, min(5, len(train_session_groups_list)))
            train_session_groups_list = train_session_groups_list[:num_test_sessions]
            print(
                f"   🧪 TEST MODE: Reduced to {len(train_session_groups_list)} training sessions.")
    except Exception as e_group:
        print(
            f"❌ Error during training set grouping: {e_group}. Check DataFrame structure.")
        sys.exit(1)

    print(
        f"\n🚀 Starting parallel training set processing ({len(train_session_groups_list)} groups)...")
    train_parallel_start_time = time.time()

    train_results_from_parallel = []
    if train_session_groups_list:
        try:
            train_results_from_parallel = Parallel(
                n_jobs=max_workers, verbose=(10 if verbose else 0), timeout=7200, prefer="processes", batch_size=1
            )(
                delayed(process_session_for_features)(
                    session_grp, DATA_ROOT / "train", bp_sos_coeffs, notch_sos_coeffs, SAMPLING_RATE,
                    num_channels_expected=expected_channels, verbose=verbose, debug=debug
                )
                for session_grp in tqdm(train_session_groups_list, desc="Train Sessions")
            )
            print(
                f"\n✅ Parallel training processing completed in {time.time() - train_parallel_start_time:.1f}s")
        except Exception as e_par_train:
            print(
                f"\n❌ ERROR in parallel training processing: {e_par_train}. Try reducing max_workers or check individual session processing.")
            # Optionally add sequential fallback here if critical

    X_train_features_list, all_train_original_indices = [], []
    for features_list, indices_list in train_results_from_parallel:
        X_train_features_list.extend(features_list)
        all_train_original_indices.extend(indices_list)

    X_train_np = np.array(
        X_train_features_list, dtype=float) if X_train_features_list else np.array([], dtype=float)
    y_train_np, train_subject_ids_np = np.array(
        [], dtype=float), np.array([], dtype=object)

    if all_train_original_indices:
        # Ensure indices are unique if there's any chance of duplicates from processing
        unique_train_indices = pd.Index(all_train_original_indices).unique()
        aligned_train_df = train_segments_for_labels_df.loc[unique_train_indices]
        y_train_np = aligned_train_df["label"].values
        if 'patient' in aligned_train_df.index.names:
            train_subject_ids_np = aligned_train_df.index.get_level_values(
                'patient').to_numpy()
        elif 'patient' in aligned_train_df.columns:
            train_subject_ids_np = aligned_train_df['patient'].to_numpy()

    if verbose:
        print(f"\n📈 Training Set Results:")
        print(
            f"   X_train shape: {X_train_np.shape if X_train_np.size > 0 else 'Empty'}")
        print(
            f"   y_train shape: {y_train_np.shape if y_train_np.size > 0 else 'Empty'}")
        print(
            f"   Train Subject IDs: {len(np.unique(train_subject_ids_np)) if train_subject_ids_np.size > 0 else 'None'}")

    # --- Test Set Feature Extraction ---
    print("\n⏳ Processing Test Set...")
    test_session_groups_list = []
    try:
        if all(level in test_segments_df.index.names for level in grouping_cols):
            test_session_groups_list = list(
                test_segments_df.groupby(level=grouping_cols))
        elif all(col in test_segments_df.columns for col in grouping_cols):
            test_session_groups_list = list(
                test_segments_df.groupby(grouping_cols))
        else:
            raise ValueError(
                f"Could not find '{grouping_cols[0]}' and '{grouping_cols[1]}' for grouping in test data.")

        if verbose:
            print(
                f"   Total test session groups: {len(test_session_groups_list)}")
        if test_mode and test_session_groups_list:
            num_test_sessions = max(1, min(5, len(test_session_groups_list)))
            test_session_groups_list = test_session_groups_list[:num_test_sessions]
            print(
                f"   🧪 TEST MODE: Reduced to {len(test_session_groups_list)} test sessions.")
    except Exception as e_group_test:
        print(
            f"❌ Error during test set grouping: {e_group_test}. Check DataFrame structure.")
        sys.exit(1)

    print(
        f"\n🚀 Starting parallel test set processing ({len(test_session_groups_list)} groups)...")
    test_parallel_start_time = time.time()

    test_results_from_parallel = []
    if test_session_groups_list:
        try:
            test_results_from_parallel = Parallel(
                n_jobs=max_workers, verbose=(10 if verbose else 0), timeout=3600, prefer="processes", batch_size=1
            )(
                delayed(process_session_for_features)(
                    session_grp, DATA_ROOT / "test", bp_sos_coeffs, notch_sos_coeffs, SAMPLING_RATE,
                    num_channels_expected=expected_channels, verbose=verbose, debug=debug
                )
                for session_grp in tqdm(test_session_groups_list, desc="Test Sessions")
            )
            print(
                f"\n✅ Parallel test processing completed in {time.time() - test_parallel_start_time:.1f}s")
        except Exception as e_par_test:
            print(
                f"\n❌ ERROR in parallel test processing: {e_par_test}. Try reducing max_workers.")

    X_test_features_list, all_test_original_indices = [], []
    for features_list, indices_list in test_results_from_parallel:
        X_test_features_list.extend(features_list)
        all_test_original_indices.extend(indices_list)

    X_test_np = np.array(
        X_test_features_list, dtype=float) if X_test_features_list else np.array([], dtype=float)

    # Align X_test to the original order of test_segments_df, handling potential MultiIndex
    if all_test_original_indices and X_test_np.size > 0:
        temp_idx_for_df = None
        original_test_index = test_segments_df.index

        if isinstance(original_test_index, pd.MultiIndex):
            try:
                temp_idx_for_df = pd.MultiIndex.from_tuples(
                    all_test_original_indices, names=original_test_index.names)
            except Exception as e_mi_create:
                if verbose:
                    print(
                        f"   Warning: Failed to create exact MultiIndex for temp_feature_df ({e_mi_create}). Fallback may occur.")
                # Fallback: try creating with default names if original names caused issue
                try:
                    num_levels = len(all_test_original_indices[0]) if all_test_original_indices and isinstance(
                        all_test_original_indices[0], tuple) else 0
                    default_names = [f'level_{i}' for i in range(
                        num_levels)] if num_levels > 0 else None
                    if default_names:
                        temp_idx_for_df = pd.MultiIndex.from_tuples(
                            all_test_original_indices, names=default_names)
                except Exception as e_mi_fallback:
                    if verbose:
                        print(
                            f"   Warning: MultiIndex fallback creation also failed ({e_mi_fallback}). Alignment might be incorrect.")
        else:  # Simple Index
            temp_idx_for_df = pd.Index(
                all_test_original_indices, name=original_test_index.name)

        if temp_idx_for_df is not None and X_test_np.ndim == 2:  # Ensure X_test_np is 2D
            # Ensure all feature vectors in X_test_np have the same length before creating DataFrame
            # This should be the case if process_session_for_features correctly uses current_session_empty_features
            # or if all signal files have the same number of channels.
            # If lengths vary, np.array(X_test_features_list) might be object type or error.
            # A robust way is to pad/truncate each feature vector in X_test_features_list to a common length
            # (e.g., expected_channels * features_per_channel) before np.array if lengths can vary.
            # For now, assume lengths are consistent or X_test_np.ndim == 2 holds.

            temp_feature_df = pd.DataFrame(X_test_np, index=temp_idx_for_df)
            # Reindex to match the original test_segments_df.index, filling missing with 0.0
            # This ensures X_test_np has one row for every row in test_segments_df, in the original order.
            try:
                aligned_X_test_df = temp_feature_df.reindex(
                    original_test_index, fill_value=0.0)
                X_test_np = aligned_X_test_df.values
                if verbose:
                    print(
                        f"   X_test aligned to original test set. New shape: {X_test_np.shape}")
            except Exception as e_reindex:
                if verbose:
                    print(
                        f"   Warning: Reindexing X_test failed ({e_reindex}). X_test might not be fully aligned.")
        elif X_test_np.ndim != 2 and X_test_np.size > 0:
            if verbose:
                print(
                    f"   Warning: X_test_np is not 2D (shape: {X_test_np.shape}). Alignment skipped. Check feature vector consistency across sessions.")
        elif temp_idx_for_df is None and verbose:
            print(
                "   Warning: Index for X_test alignment could not be created. X_test not aligned.")

    if verbose:
        print(f"\n🧪 Test Set Results:")
        print(
            f"   X_test shape: {X_test_np.shape if X_test_np.size > 0 else 'Empty'}")

    # --- Final Checks and Save ---
    print("\n--- Final Dataset Shapes: ---")
    print(f"X_train: {X_train_np.shape if X_train_np.size > 0 else 'Empty'}")
    print(f"y_train: {y_train_np.shape if y_train_np.size > 0 else 'Empty'}")
    print(f"X_test: {X_test_np.shape if X_test_np.size > 0 else 'Empty'}")
    print(
        f"Train Subject IDs: {train_subject_ids_np.shape if train_subject_ids_np.size > 0 else 'Empty'}")

    print("\n💾 Saving extracted feature arrays...")
    output_features_path = DATA_ROOT / "extracted_features"
    output_labels_path = DATA_ROOT / "labels"

    if X_test_np.size > 0:
        np.save(output_features_path / "X_test.npy", X_test_np)
    else:
        print("   X_test is empty, not saving.")
    if X_train_np.size > 0:
        np.save(output_features_path / "X_train.npy", X_train_np)
    else:
        print("   X_train is empty, not saving.")
    if y_train_np.size > 0:
        np.save(output_labels_path / "y_train.npy", y_train_np)
    else:
        print("   y_train is empty, not saving.")
    if train_subject_ids_np.size > 0:
        np.save(output_features_path /
                "sample_subject_array_train.npy", train_subject_ids_np)
    else:
        print("   sample_subject_list_train is empty, not saving.")

    print(
        f"\n✅ Feature extraction and saving complete to {DATA_ROOT.resolve()}.")
    print(
        f"⏰ Total pipeline execution time: {time.time() - overall_pipeline_start_time:.2f} seconds.")


if __name__ == "__main__":
    run_verbose = True
    run_test_mode = False
    run_max_workers = max(1, multiprocessing.cpu_count() - 2)

    # Example: For a full run with verbose logging
    main(verbose=True, test_mode=False, max_workers=run_max_workers)
