import os
import sys
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed  # For parallel processing
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch, butter, iirnotch, sosfiltfilt, tf2sos, hilbert
from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq

from src.utils.signal import spectral_entropy
from src.utils.dataset import ensure_eeg_multiindex

def _hjorth_parameters(signal: np.ndarray) -> Tuple[float, float]:
    """Calculate Hjorth mobility and complexity parameters."""
    dx = np.diff(signal)
    var_x = np.var(signal)
    mobility, complexity = 0.0, 0.0
    
    if var_x > 1e-10:
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (var_x + 1e-12))
        
        if var_dx > 1e-10:
            ddx = np.diff(dx)
            complexity = np.sqrt(np.var(ddx) / (var_dx + 1e-12))
    
    return mobility, complexity


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Calculate zero crossing rate."""
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def _sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Calculate sample entropy (approximate version for efficiency)."""
    try:
        N = len(signal)
        if N < m + 1:
            return 0.0
        
        # Normalize signal
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        r = r * np.std(signal)
        
        def _maxdist(x, y, m):
            return max([abs(ua - va) for ua, va in zip(x, y)])
        
        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append([signal[i + j] for j in range(m)])
            
            count = 0
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j], m) <= r:
                        count += 1
            
            return count / (N - m + 1) / (N - m) * 2
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m == 0 or phi_m1 == 0:
            return 0.0
        
        return -np.log(phi_m1 / phi_m)
    except:
        return 0.0


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, percentage: float = 95) -> float:
    """Calculate spectral edge frequency (frequency below which X% of power lies)."""
    total_power = np.sum(psd)
    if total_power == 0:
        return 0.0
    
    cumsum_power = np.cumsum(psd)
    threshold = total_power * percentage / 100
    idx = np.where(cumsum_power >= threshold)[0]
    
    return freqs[idx[0]] if len(idx) > 0 else freqs[-1]


def _peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Find the frequency with maximum power."""
    if len(psd) == 0:
        return 0.0
    peak_idx = np.argmax(psd)
    return freqs[peak_idx]


def _extract_channel_features(channel_signal: np.ndarray, fs: int,
                              bands: Dict[str, Tuple[float, float]]) -> List[float]:
    """Extract comprehensive EEG features from a single channel."""
    
    # Handle edge cases
    if len(channel_signal) < 10:
        return [0.0] * 35  # Return zeros for all features
    
    # === TIME DOMAIN FEATURES ===
    
    # Basic statistical features
    mean_val = np.mean(channel_signal)
    std_val = np.std(channel_signal)
    variance = np.var(channel_signal)
    rms_val = np.sqrt(np.mean(channel_signal**2))
    min_val = np.min(channel_signal)
    max_val = np.max(channel_signal)
    peak_to_peak = max_val - min_val
    
    # Higher order statistics
    skewness = skew(channel_signal) if len(channel_signal) > 1 else 0.0
    kurt = kurtosis(channel_signal) if len(channel_signal) > 1 else 0.0
    
    # Activity measures
    line_length = np.sum(np.abs(np.diff(channel_signal)))
    zero_cross_rate = _zero_crossing_rate(channel_signal)
    
    # Hjorth parameters
    hj_mob, hj_cmp = _hjorth_parameters(channel_signal)
    
    # Nonlinear features
    sample_ent = _sample_entropy(channel_signal)
    
    # === FREQUENCY DOMAIN FEATURES ===
    
    nperseg_welch = min(len(channel_signal), 2 * fs)
    
    # Initialize spectral features
    bp_values = {name: 0.0 for name in bands}
    rel_powers = {name: 0.0 for name in bands}
    spec_ent_val = 0.0
    spectral_centroid = 0.0
    spectral_rolloff = 0.0
    peak_freq = 0.0
    spec_edge_95 = 0.0
    
    # Band ratios
    alpha_beta_ratio = 0.0
    theta_alpha_ratio = 0.0
    delta_theta_ratio = 0.0
    
    if len(channel_signal) >= 2 and nperseg_welch >= 2:
        try:
            freqs, psd = welch(channel_signal, fs=fs, nperseg=nperseg_welch)
            total_pow = np.sum(psd) + 1e-12
            
            # Band powers
            for name, (lo, hi) in bands.items():
                idx = (freqs >= lo) & (freqs < hi)
                bp_values[name] = np.trapezoid(psd[idx], freqs[idx]) if np.any(idx) else 0.0
                rel_powers[name] = bp_values[name] / total_pow
            
            # Spectral features
            if total_pow > 1e-10:
                spec_ent_val = spectral_entropy(psd)
                
                # Spectral centroid (weighted mean frequency)
                spectral_centroid = np.sum(freqs * psd) / total_pow
                
                # Spectral rolloff (95% energy frequency)
                spectral_rolloff = _spectral_edge_frequency(freqs, psd, 95)
                
                # Peak frequency
                peak_freq = _peak_frequency(freqs, psd)
                
                # Spectral edge frequency
                spec_edge_95 = _spectral_edge_frequency(freqs, psd, 95)
            
            # Band ratios (common in EEG analysis)
            if bp_values.get("beta", 0.0) > 1e-12:
                alpha_beta_ratio = bp_values.get("alpha", 0.0) / bp_values.get("beta", 0.0)
            if bp_values.get("alpha", 0.0) > 1e-12:
                theta_alpha_ratio = bp_values.get("theta", 0.0) / bp_values.get("alpha", 0.0)
            if bp_values.get("theta", 0.0) > 1e-12:
                delta_theta_ratio = bp_values.get("delta", 0.0) / bp_values.get("theta", 0.0)
                
        except (ValueError, RuntimeError, ZeroDivisionError):
            pass
    
    # Compile all features (35 total)
    features = [
        # Time domain features (14)
        mean_val, std_val, variance, rms_val, min_val, max_val, peak_to_peak,
        skewness, kurt, line_length, zero_cross_rate, hj_mob, hj_cmp, sample_ent,
        
        # Absolute band powers (5)
        bp_values.get("delta", 0.0), bp_values.get("theta", 0.0), 
        bp_values.get("alpha", 0.0), bp_values.get("beta", 0.0), bp_values.get("gamma", 0.0),
        
        # Relative band powers (5)
        rel_powers.get("delta", 0.0), rel_powers.get("theta", 0.0),
        rel_powers.get("alpha", 0.0), rel_powers.get("beta", 0.0), rel_powers.get("gamma", 0.0),
        
        # Spectral features (6)
        spec_ent_val, spectral_centroid, spectral_rolloff, peak_freq, spec_edge_95,
        
        # Band ratios (3)
        alpha_beta_ratio, theta_alpha_ratio, delta_theta_ratio,
        
        # Additional spectral feature (1)
        np.sum(list(bp_values.values()))  # Total power
    ]
    
    return features


def _extract_segment_features(segment_signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """Extract features from all channels in a segment."""
    all_channel_features: List[float] = []
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 50)}
    num_channels = segment_signal.shape[1]
    
    for ch_idx in range(num_channels):
        ch_feats = _extract_channel_features(segment_signal[:, ch_idx], fs, bands)
        all_channel_features.extend(ch_feats)
    
    features_array = np.asarray(all_channel_features, dtype=float)
    return np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)


def process_session_for_features(
    session_group: Tuple[Any, pd.DataFrame],
    base_signal_path: Path,
    bp_filter_coeffs: np.ndarray,
    notch_filter_coeffs: Optional[np.ndarray],
    f_s: int = 250,
    num_channels: int = 19,
    num_features_per_channel: int = 35,
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[Any]]:
    group_name, session_df = session_group
    
    if verbose:
        print(f"🔄 Starting job for session group: {group_name}")
        print(f"   📊 Session has {len(session_df)} segments to process")
        print(f"   🗂️  Expected features per segment: {num_channels * num_features_per_channel}")
    else:
        print(f"Processing session: {group_name} ({len(session_df)} segments)")

    # Calculate expected feature size
    expected_feature_size = num_channels * num_features_per_channel
    empty_features = np.zeros(expected_feature_size, dtype=float)

    features_list_for_session: List[np.ndarray] = []
    all_original_indices: List[Any] = []

    if 'signals_path' not in session_df.columns or session_df.empty:
        if verbose:
            print(f"   ⚠️  No signals_path column or empty session for {group_name}")
        # Return empty features for all rows in this session
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        return features_list_for_session, all_original_indices

    relative_signal_file = session_df['signals_path'].iloc[0]
    full_session_signal_path = base_signal_path / relative_signal_file
    
    if verbose:
        print(f"   📁 Loading signal file: {relative_signal_file}")
        print(f"   📍 Full path: {full_session_signal_path}")

    if not full_session_signal_path.exists():
        if verbose:
            print(f"   ❌ Signal file not found: {full_session_signal_path}")
        # Return empty features for all rows in this session
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        return features_list_for_session, all_original_indices

    try:
        if verbose:
            print(f"   🔄 Reading parquet file...")
        session_signal_df = pd.read_parquet(full_session_signal_path)
        session_signal_values = session_signal_df.values
        if verbose:
            print(f"   ✅ Signal loaded: shape {session_signal_values.shape}")
    except Exception as e:
        print(
            f"Error loading session signal {full_session_signal_path} for group {group_name}: {e}. Using empty features for all segments.")
        # Return empty features for all rows in this session
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        return features_list_for_session, all_original_indices

    if verbose:
        print(f"   🔧 Applying filters...")
        print(f"      Original signal range: [{np.min(session_signal_values):.4f}, {np.max(session_signal_values):.4f}]")
    
    processed_signal = sosfiltfilt(
        bp_filter_coeffs, session_signal_values, axis=0)
    if notch_filter_coeffs is not None:
        processed_signal = sosfiltfilt(
            notch_filter_coeffs, processed_signal, axis=0)

    if verbose:
        print(f"      Filtered signal range: [{np.min(processed_signal):.4f}, {np.max(processed_signal):.4f}]")
        print(f"   🔄 Applying re-referencing and normalization...")

    avg_reference = np.mean(processed_signal, axis=1, keepdims=True)
    processed_signal = processed_signal - avg_reference
    mean_per_channel = np.mean(processed_signal, axis=0, keepdims=True)
    std_per_channel = np.std(processed_signal, axis=0, keepdims=True)
    processed_signal = (processed_signal - mean_per_channel) / \
        (std_per_channel + 1e-6)
    
    if verbose:
        print(f"      Normalized signal range: [{np.min(processed_signal):.4f}, {np.max(processed_signal):.4f}]")
        print(f"   🎯 Processing {len(session_df)} segments...")

    segment_count = 0
    valid_segments = 0
    failed_segments = 0

    for original_idx, row in session_df.iterrows():
        segment_count += 1
        
        if verbose and segment_count % 10 == 0:
            print(f"      Progress: {segment_count}/{len(session_df)} segments processed")
        elif not verbose:
            print(f"Processing segment {segment_count}/{len(session_df)} for original index: {original_idx}")
            
        try:
            t0 = int(row["start_time"] * f_s)
            tf = int(row["end_time"] * f_s)
            max_len = processed_signal.shape[0]
            segment_valid = True
            
            if verbose and segment_count <= 3:  # Show details for first few segments
                print(f"         Segment {segment_count}: t0={t0}, tf={tf}, max_len={max_len}")

            if t0 < 0 or t0 >= max_len:
                segment_valid = False
                if verbose and segment_count <= 3:
                    print(f"         Invalid: t0 out of bounds")
            elif tf > max_len:
                tf = max_len
                if t0 >= tf:
                    segment_valid = False
                    if verbose and segment_count <= 3:
                        print(f"         Invalid: tf adjusted but still invalid")
            elif t0 >= tf:
                segment_valid = False
                if verbose and segment_count <= 3:
                    print(f"         Invalid: t0 >= tf")

            if not segment_valid:
                features_list_for_session.append(empty_features.copy())
                all_original_indices.append(original_idx)
                failed_segments += 1
                continue

            segment = processed_signal[t0:tf, :]
            if segment.shape[0] < 2:
                features_list_for_session.append(empty_features.copy())
                all_original_indices.append(original_idx)
                failed_segments += 1
                if verbose and segment_count <= 3:
                    print(f"         Invalid: segment too short ({segment.shape[0]} samples)")
                continue

            if verbose and segment_count <= 3:
                print(f"         Extracting features from segment shape: {segment.shape}")
                
            features = _extract_segment_features(segment, fs=f_s)
            features_list_for_session.append(features)
            all_original_indices.append(original_idx)
            valid_segments += 1
            
            if verbose and segment_count <= 3:
                print(f"         Features extracted: {len(features)} values")
                print(f"         Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
                
        except Exception as e:
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
            failed_segments += 1
            if verbose:
                print(f"         ❌ Error processing segment {segment_count}: {e}")
            continue
    
    if verbose:
        print(f"   ✅ Session {group_name} completed:")
        print(f"      Total segments: {segment_count}")
        print(f"      Valid segments: {valid_segments}")
        print(f"      Failed segments: {failed_segments}")
        print(f"      Success rate: {valid_segments/segment_count*100:.1f}%")
    else:
        print(f"Job completed for session group: {group_name} - {valid_segments}/{segment_count} valid segments")
        
    return features_list_for_session, all_original_indices


IS_SCITAS = False
CPU_COUNT = multiprocessing.cpu_count()

def main(verbose: bool = False):
    LOCAL_DATA_ROOT = Path("data")
    (LOCAL_DATA_ROOT / "extracted_features").mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    
    DATA_ROOT= Path("/home/ogut/data") if IS_SCITAS else Path("data")
    print(f"🔍 Looking for data in: {LOCAL_DATA_ROOT.resolve()}")
    
    if verbose:
        print(f"\n📊 ENHANCED EEG FEATURE EXTRACTION")
        print(f"{'='*50}")
        print(f"🧠 Features per channel: 35 (vs. original 12)")
        print(f"📈 Feature categories:")
        print(f"   • Time Domain: 14 features")
        print(f"   • Absolute Band Powers: 5 features")
        print(f"   • Relative Band Powers: 5 features")
        print(f"   • Spectral Features: 5 features")
        print(f"   • Band Ratios: 3 features")
        print(f"   • Additional: 3 features")
        print(f"{'='*50}\n")

    try:
        clips_tr_full = pd.read_parquet(
            DATA_ROOT / "train" / "segments.parquet")
        clips_te_full = pd.read_parquet(
            DATA_ROOT / "test" / "segments.parquet")
        
        if verbose:
            print(f"✅ Successfully loaded dataset files:")
            print(f"   📋 Training segments: {clips_tr_full.shape}")
            print(f"   📋 Test segments: {clips_te_full.shape}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: Parquet file not found. Details: {e}")
        sys.exit(1)
    
    print(clips_tr_full.shape)

    # --- Ensure MultiIndex ---
    clips_tr_full = ensure_eeg_multiindex(clips_tr_full, id_col_name='id')
    clips_te_full = ensure_eeg_multiindex(clips_te_full, id_col_name='id')

    # Filter out rows with NaN labels in training set
    print("🔄 Filtering training set for valid labels...")
    clips_tr_for_labels = clips_tr_full[~clips_tr_full.label.isna()].copy()
    clips_te = clips_te_full.copy()
    
    if verbose:
        print(f"   📊 Training set after label filtering: {clips_tr_for_labels.shape}")
        print(f"   📊 Test set: {clips_te.shape}")
        print(f"   🏷️  Unique labels in training: {clips_tr_for_labels['label'].nunique()}")
        label_dist = clips_tr_for_labels['label'].value_counts().sort_index()
        print(f"   📈 Label distribution:")
        for label, count in label_dist.items():
            print(f"      Label {label}: {count} samples ({count/len(clips_tr_for_labels)*100:.1f}%)")

    F_S = 250
    BP_FILTER_FREQS = (0.5, 50.0)
    NOTCH_FREQ_HZ = 60.0
    NOTCH_Q_FACTOR = 30.0

    print(
        f"\n🔧 Designing filters: BP={BP_FILTER_FREQS}Hz, Notch={NOTCH_FREQ_HZ}Hz, FS={F_S}Hz")
    
    if verbose:
        print(f"   🌊 Bandpass filter: {BP_FILTER_FREQS[0]}-{BP_FILTER_FREQS[1]} Hz")
        print(f"   🚫 Notch filter: {NOTCH_FREQ_HZ} Hz (Q={NOTCH_Q_FACTOR})")
        print(f"   📡 Sampling rate: {F_S} Hz")
        
    bp_filter_coeffs_sos = butter(
        4, BP_FILTER_FREQS, btype="bandpass", output="sos", fs=F_S)
    notch_filter_coeffs_ba = iirnotch(
        w0=NOTCH_FREQ_HZ, Q=NOTCH_Q_FACTOR, fs=F_S)
    notch_filter_coeffs_sos = tf2sos(*notch_filter_coeffs_ba)

    # --- Training Set Feature Extraction ---
    print("\n⏳ Processing Training Set...")
    
    if verbose:
        print(f"🧬 Feature extraction details:")
        print(f"   🔬 Features per channel: 35")
        print(f"   📡 Expected channels per segment: 19")
        print(f"   🎯 Total features per segment: 19 × 35 = 665")
        print(f"   💻 Parallel processing: enabled ({CPU_COUNT} cores)")

    train_session_groups = []
    try:
        # Prefer grouping by index levels if they exist as per user's description
        if all(level in clips_tr_for_labels.index.names for level in ["patient", "session"]):
            if verbose:
                print(f"   📊 Grouping by index levels: ['patient', 'session']")
            train_session_groups = list(
                clips_tr_for_labels.groupby(level=["patient", "session"]))
        # Fallback to columns if index levels are not set as expected
        elif all(col in clips_tr_for_labels.columns for col in ["patient", "session"]):
            if verbose:
                print(f"   📊 Grouping by columns: ['patient', 'session']")
            train_session_groups = list(
                clips_tr_for_labels.groupby(["patient", "session"]))
        else:
            raise ValueError(
                "Could not find 'patient' and 'session' as index levels or columns in clips_tr_for_labels for grouping.")
                
        if verbose:
            print(f"   🗂️  Total session groups: {len(train_session_groups)}")
            
    except Exception as e:
        print(f"❌ Error during groupby for training set: {e}")
        print(f"clips_tr_for_labels index: {clips_tr_for_labels.index.names}")
        print(f"clips_tr_for_labels columns: {clips_tr_for_labels.columns}")
        sys.exit(1)

    train_processing_results = Parallel(n_jobs=CPU_COUNT)(
        delayed(process_session_for_features)(
            session_group,
            base_signal_path=LOCAL_DATA_ROOT / "train",
            bp_filter_coeffs=bp_filter_coeffs_sos,
            notch_filter_coeffs=notch_filter_coeffs_sos,
            f_s=F_S,
            verbose=verbose
        )
        for session_group in tqdm(train_session_groups, desc="Processing Train Sessions")
    )

    if verbose:
        print(f"\n🔄 Collecting training results from {len(train_processing_results)} jobs...")
        
    X_train_list = []
    all_train_indices = []
    total_segments_processed = 0
    total_valid_segments = 0
    
    for i, (session_feat_list, session_indices_list) in enumerate(train_processing_results):
        if verbose and i % 10 == 0:
            print(f"   Processing job result {i+1}/{len(train_processing_results)}...")
            
        if session_feat_list:
            X_train_list.extend(session_feat_list)
            all_train_indices.extend(session_indices_list)
            total_valid_segments += len(session_feat_list)
        total_segments_processed += len(session_indices_list) if session_indices_list else 0

    if verbose:
        print(f"   ✅ Training data collection complete:")
        print(f"      Total segments processed: {total_segments_processed}")
        print(f"      Valid segments: {total_valid_segments}")
        print(f"      Success rate: {total_valid_segments/total_segments_processed*100:.1f}%" if total_segments_processed > 0 else "      Success rate: 0%")

    X_train = np.array(X_train_list) if X_train_list else np.array([])

    y_train = np.array([])
    sample_subject_list_train = np.array([])
    if all_train_indices:
        # Use .loc with the collected indices on the DataFrame that was used for iteration (clips_tr_for_labels)
        clips_tr_aligned = clips_tr_for_labels.loc[all_train_indices]
        y_train = clips_tr_aligned["label"].values

        if 'patient' in clips_tr_aligned.index.names:
            sample_subject_list_train = clips_tr_aligned.index.get_level_values(
                'patient').to_numpy()
        elif 'patient' in clips_tr_aligned.columns:
            sample_subject_list_train = clips_tr_aligned['patient'].to_numpy()
    else:
        print("⚠️  Warning: No training features were extracted. y_train and sample_subject_list_train will be empty.")
        
    if verbose:
        print(f"\n📈 Training set feature extraction results:")
        print(f"   ✅ Features extracted: {X_train.shape if X_train.size > 0 else 'None'}")
        print(f"   🏷️  Labels extracted: {y_train.shape if y_train.size > 0 else 'None'}")
        print(f"   👤 Subjects: {len(np.unique(sample_subject_list_train)) if sample_subject_list_train.size > 0 else 'None'}")
        if X_train.size > 0:
            features_per_channel = X_train.shape[1] // 19
            print(f"   🔬 Actual features per channel: {features_per_channel}")
            print(f"   📊 Feature statistics:")
            print(f"      Mean: {np.mean(X_train):.4f}")
            print(f"      Std: {np.std(X_train):.4f}")
            print(f"      Min: {np.min(X_train):.4f}")
            print(f"      Max: {np.max(X_train):.4f}")
            non_zero_features = np.count_nonzero(X_train, axis=0)
            print(f"   🎯 Non-zero features: {np.sum(non_zero_features > 0)}/{len(non_zero_features)}")
        
    if verbose:
        print(f"\n📈 Training set feature extraction results:")
        print(f"   ✅ Features extracted: {X_train.shape if X_train.size > 0 else 'None'}")
        print(f"   🏷️  Labels extracted: {y_train.shape if y_train.size > 0 else 'None'}")
        print(f"   👤 Subjects: {len(np.unique(sample_subject_list_train)) if sample_subject_list_train.size > 0 else 'None'}")
        if X_train.size > 0:
            features_per_channel = X_train.shape[1] // 19
            print(f"   🔬 Actual features per channel: {features_per_channel}")
            print(f"   📊 Feature statistics:")
            print(f"      Mean: {np.mean(X_train):.4f}")
            print(f"      Std: {np.std(X_train):.4f}")
            print(f"      Min: {np.min(X_train):.4f}")
            print(f"      Max: {np.max(X_train):.4f}")
            non_zero_features = np.count_nonzero(X_train, axis=0)
            print(f"   🎯 Non-zero features: {np.sum(non_zero_features > 0)}/{len(non_zero_features)}")

    # --- Test Set Feature Extraction ---
    print("\n⏳ Processing Test Set...")
    
    if verbose:
        print(f"🧪 Test set processing:")
        
    test_session_groups = []
    try:
        if all(level in clips_te.index.names for level in ["patient", "session"]):
            if verbose:
                print(f"   📊 Grouping by index levels: ['patient', 'session']")
            test_session_groups = list(
                clips_te.groupby(level=["patient", "session"]))
        elif all(col in clips_te.columns for col in ["patient", "session"]):
            if verbose:
                print(f"   📊 Grouping by columns: ['patient', 'session']")
            test_session_groups = list(
                clips_te.groupby(["patient", "session"]))
        else:
            # If test set doesn't have patient/session, how should it be grouped for signals_path?
            # This might require a different strategy if signals_path is not per-segment.
            # For now, assume it also has a structure that allows similar grouping.
            raise ValueError(
                "Could not find 'patient' and 'session' as index levels or columns in clips_te for grouping.")
                
        if verbose:
            print(f"   🗂️  Total test session groups: {len(test_session_groups)}")
            
    except Exception as e:
        print(f"❌ Error during groupby for test set: {e}")
        print(f"clips_te index: {clips_te.index.names}")
        print(f"clips_te columns: {clips_te.columns}")
        sys.exit(1)

    test_processing_results = Parallel(n_jobs=-1)(
        delayed(process_session_for_features)(
            session_group,
            base_signal_path=LOCAL_DATA_ROOT / "test",
            bp_filter_coeffs=bp_filter_coeffs_sos,
            notch_filter_coeffs=notch_filter_coeffs_sos,
            f_s=F_S,
            verbose=verbose
        )
        for session_group in tqdm(test_session_groups, desc="Processing Test Sessions")
    )

    if verbose:
        print(f"\n🔄 Collecting test results from {len(test_processing_results)} jobs...")

    X_test_list = []
    total_test_segments = 0
    valid_test_segments = 0
    
    # For test set, we collect all features (including empty ones for failed processing)
    for i, (session_feat_list, session_indices_list) in enumerate(test_processing_results):
        if verbose and i % 10 == 0:
            print(f"   Processing test job result {i+1}/{len(test_processing_results)}...")
            
        if session_feat_list:
            X_test_list.extend(session_feat_list)
            valid_test_segments += len(session_feat_list)
        total_test_segments += len(session_indices_list) if session_indices_list else 0
        
    X_test = np.array(X_test_list) if X_test_list else np.array([])
    
    if verbose:
        print(f"   ✅ Test data collection complete:")
        print(f"      Total test segments processed: {total_test_segments}")
        print(f"      Valid test segments: {valid_test_segments}")
        print(f"      Test success rate: {valid_test_segments/total_test_segments*100:.1f}%" if total_test_segments > 0 else "      Test success rate: 0%")
        print(f"   🧪 Test set feature extraction results:")
        print(f"      Features extracted: {X_test.shape if X_test.size > 0 else 'None'}")
        if X_test.size > 0:
            print(f"      Feature statistics:")
            print(f"         Mean: {np.mean(X_test):.4f}")
            print(f"         Std: {np.std(X_test):.4f}")
            print(f"         Min: {np.min(X_test):.4f}")
            print(f"         Max: {np.max(X_test):.4f}")

    # --- Check correct values ---
    print("\n--- Final dataset shapes: ---")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("sample_subject_list_train:", sample_subject_list_train.shape)

    # --- Save arrays ---
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

if __name__ == "__main__":
    main(verbose=True)