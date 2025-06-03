#!/usr/bin/env python3
"""
Simplified EEG Feature Extraction for Seizure Detection

This script extracts only the most important and medically-proven features
for seizure detection based on state-of-the-art research:
- Band powers (absolute and relative)
- Hjorth parameters (mobility, complexity)
- Spectral entropy
- Line length
- Key band ratios

Features: 12 per channel × 19 channels = 228 total features
(Down from 35 per channel × 19 channels = 665 features)
"""

import os
import time
import multiprocessing
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch, butter, iirnotch, sosfiltfilt, tf2sos

from src.utils.signal import spectral_entropy
from src.utils.dataset import ensure_eeg_multiindex


def extract_seizure_features(channel_signal: np.ndarray, fs: int = 250) -> List[float]:
    """
    Extract the most important features for seizure detection from a single channel.
    
    Returns 12 features per channel:
    - 5 band powers (delta, theta, alpha, beta, gamma)
    - 2 relative powers (alpha, theta)
    - 2 Hjorth parameters (mobility, complexity)
    - 1 spectral entropy
    - 1 line length
    - 1 theta/alpha ratio
    """
    # Handle edge cases
    if len(channel_signal) < 10:
        return [0.0] * 12
    
    # Time domain features
    line_length = np.sum(np.abs(np.diff(channel_signal)))
    
    # Hjorth parameters
    dx = np.diff(channel_signal)
    var_x = np.var(channel_signal)
    mobility = complexity = 0.0
    
    if var_x > 1e-10:
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (var_x + 1e-12))
        
        if var_dx > 1e-10:
            ddx = np.diff(dx)
            complexity = np.sqrt(np.var(ddx) / (var_dx + 1e-12))
    
    # Frequency domain features
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }
    
    # Initialize band powers
    bp = {name: 0.0 for name in bands}
    rel_alpha = rel_theta = theta_alpha_ratio = spec_ent = 0.0
    
    # Calculate power spectral density
    try:
        nperseg = min(len(channel_signal), 2 * fs)
        if nperseg >= 2:
            freqs, psd = welch(channel_signal, fs=fs, nperseg=nperseg)
            total_power = np.sum(psd) + 1e-12
            
            # Band powers
            for name, (lo, hi) in bands.items():
                idx = (freqs >= lo) & (freqs < hi)
                bp[name] = np.trapezoid(psd[idx], freqs[idx]) if np.any(idx) else 0.0
            
            # Relative powers (most important for seizure detection)
            rel_alpha = bp["alpha"] / total_power
            rel_theta = bp["theta"] / total_power
            
            # Theta/alpha ratio (key seizure indicator)
            theta_alpha_ratio = bp["theta"] / (bp["alpha"] + 1e-12)
            
            # Spectral entropy
            spec_ent = spectral_entropy(psd) if len(psd) > 0 else 0.0
            
    except (ValueError, RuntimeError):
        pass
    
    # Return 12 features
    return [
        # Band powers (5 features)
        bp["delta"], bp["theta"], bp["alpha"], bp["beta"], bp["gamma"],
        # Relative powers (2 features) - critical for seizure detection
        rel_alpha, rel_theta,
        # Hjorth parameters (2 features) - measure signal complexity
        mobility, complexity,
        # Spectral entropy (1 feature) - measures frequency distribution
        spec_ent,
        # Line length (1 feature) - measures signal activity
        line_length,
        # Theta/alpha ratio (1 feature) - key seizure biomarker
        theta_alpha_ratio
    ]


def extract_segment_features(segment_signal: np.ndarray, fs: int = 250) -> np.ndarray:
    """Extract features from all channels in a segment."""
    all_features = []
    num_channels = segment_signal.shape[1]
    
    for ch_idx in range(num_channels):
        ch_features = extract_seizure_features(segment_signal[:, ch_idx], fs)
        all_features.extend(ch_features)
    
    # Convert to array and handle NaN/inf values
    features_array = np.asarray(all_features, dtype=float)
    return np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)


def process_session(session_group: Tuple[Any, pd.DataFrame], 
                   base_signal_path: Path,
                   bp_filter_coeffs: np.ndarray,
                   notch_filter_coeffs: Optional[np.ndarray],
                   fs: int = 250) -> Tuple[List[np.ndarray], List[Any]]:
    """Process a single session for feature extraction."""
    
    group_name, session_df = session_group
    start_time = time.time()
    
    # Expected feature size: 12 features per channel × 19 channels = 228
    expected_feature_size = 19 * 12
    empty_features = np.zeros(expected_feature_size, dtype=float)
    
    features_list = []
    indices_list = []
    
    # Check if session has signal path
    if 'signals_path' not in session_df.columns or session_df.empty:
        print(f"   ⚠️  No signals_path for {group_name}")
        for original_idx, _ in session_df.iterrows():
            features_list.append(empty_features.copy())
            indices_list.append(original_idx)
        return features_list, indices_list
    
    # Load signal file
    relative_signal_file = session_df['signals_path'].iloc[0]
    full_session_signal_path = base_signal_path / relative_signal_file
    
    if not full_session_signal_path.exists():
        print(f"   ❌ File not found: {group_name}")
        for original_idx, _ in session_df.iterrows():
            features_list.append(empty_features.copy())
            indices_list.append(original_idx)
        return features_list, indices_list
    
    try:
        # Load and preprocess signal
        session_signal_df = pd.read_parquet(full_session_signal_path)
        session_signal_values = session_signal_df.values
        
        # Apply filters
        processed_signal = sosfiltfilt(bp_filter_coeffs, session_signal_values, axis=0)
        if notch_filter_coeffs is not None:
            processed_signal = sosfiltfilt(notch_filter_coeffs, processed_signal, axis=0)
        
        # Re-reference to average
        avg_reference = np.mean(processed_signal, axis=1, keepdims=True)
        processed_signal = processed_signal - avg_reference
        
        # Z-score normalization per channel
        mean_per_channel = np.mean(processed_signal, axis=0, keepdims=True)
        std_per_channel = np.std(processed_signal, axis=0, keepdims=True)
        processed_signal = (processed_signal - mean_per_channel) / (std_per_channel + 1e-6)
        
        # Process each segment
        valid_segments = 0
        for original_idx, row in session_df.iterrows():
            try:
                t0 = int(row["start_time"] * fs)
                tf = int(row["end_time"] * fs)
                max_len = processed_signal.shape[0]
                
                # Validate segment bounds
                if t0 < 0 or t0 >= max_len or tf <= t0 or tf > max_len:
                    features_list.append(empty_features.copy())
                    indices_list.append(original_idx)
                    continue
                
                # Extract segment
                segment = processed_signal[t0:tf, :]
                if segment.shape[0] < 2:
                    features_list.append(empty_features.copy())
                    indices_list.append(original_idx)
                    continue
                
                # Extract features
                features = extract_segment_features(segment, fs=fs)
                features_list.append(features)
                indices_list.append(original_idx)
                valid_segments += 1
                
            except Exception as e:
                features_list.append(empty_features.copy())
                indices_list.append(original_idx)
                continue
        
        elapsed = time.time() - start_time
        print(f"   ✅ {group_name}: {valid_segments}/{len(session_df)} valid ({elapsed:.1f}s)")
        
    except Exception as e:
        print(f"   ❌ Error processing {group_name}: {e}")
        for original_idx, _ in session_df.iterrows():
            features_list.append(empty_features.copy())
            indices_list.append(original_idx)
    
    return features_list, indices_list


def main(verbose: bool = False, test_mode: bool = False, max_workers: int = None):
    """Main function for simplified feature extraction."""
    
    print(f"\n🧠 SIMPLIFIED EEG FEATURE EXTRACTION FOR SEIZURE DETECTION")
    print(f"{'='*65}")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set reasonable defaults for parallel processing
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 4)
    
    print(f"🔧 Configuration: test_mode={test_mode}, max_workers={max_workers}")
    print(f"📊 Features: 12 per channel × 19 channels = 228 total")
    print(f"   • Band Powers: 5 features (delta, theta, alpha, beta, gamma)")
    print(f"   • Relative Powers: 2 features (alpha, theta)")
    print(f"   • Hjorth Parameters: 2 features (mobility, complexity)")
    print(f"   • Spectral Entropy: 1 feature")
    print(f"   • Line Length: 1 feature")
    print(f"   • Theta/Alpha Ratio: 1 feature")
    
    # Setup paths
    LOCAL_DATA_ROOT = Path("data")
    (LOCAL_DATA_ROOT / "extracted_features").mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    
    DATA_ROOT = Path("data")  # Simplified for local use
    
    try:
        # Load data
        clips_tr_full = pd.read_parquet(DATA_ROOT / "train" / "segments.parquet")
        clips_te_full = pd.read_parquet(DATA_ROOT / "test" / "segments.parquet")
        print(f"✅ Loaded training: {clips_tr_full.shape}, test: {clips_te_full.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Ensure MultiIndex
    clips_tr_full = ensure_eeg_multiindex(clips_tr_full, id_col_name='id')
    clips_te_full = ensure_eeg_multiindex(clips_te_full, id_col_name='id')
    
    # Filter training data
    clips_tr = clips_tr_full[~clips_tr_full.label.isna()].copy()
    clips_te = clips_te_full.copy()
    
    print(f"📊 After filtering - Training: {clips_tr.shape}, Test: {clips_te.shape}")
    
    # Setup filters
    F_S = 250
    BP_FILTER_FREQS = (0.5, 50.0)
    NOTCH_FREQ_HZ = 60.0
    NOTCH_Q_FACTOR = 30.0
    
    bp_filter_coeffs = butter(4, BP_FILTER_FREQS, btype="bandpass", output="sos", fs=F_S)
    notch_filter_coeffs_ba = iirnotch(w0=NOTCH_FREQ_HZ, Q=NOTCH_Q_FACTOR, fs=F_S)
    notch_filter_coeffs = tf2sos(*notch_filter_coeffs_ba)
    
    # Process training data
    print(f"\n⏳ Processing Training Set...")
    
    train_session_groups = list(clips_tr.groupby(level=["patient", "session"]))
    if test_mode:
        train_session_groups = train_session_groups[:3]
        print(f"🧪 TEST MODE: Processing only {len(train_session_groups)} sessions")
    
    print(f"🚀 Processing {len(train_session_groups)} training sessions...")
    
    # Use joblib for parallel processing
    from joblib import Parallel, delayed
    
    train_results = Parallel(n_jobs=max_workers, verbose=1)(
        delayed(process_session)(
            session_group,
            DATA_ROOT / "train",
            bp_filter_coeffs,
            notch_filter_coeffs,
            F_S
        ) for session_group in tqdm(train_session_groups, desc="Training Sessions")
    )
    
    # Collect training results
    X_train_list = []
    all_train_indices = []
    
    for features_list, indices_list in train_results:
        X_train_list.extend(features_list)
        all_train_indices.extend(indices_list)
    
    X_train = np.array(X_train_list) if X_train_list else np.array([])
    
    # Extract labels
    y_train = np.array([])
    sample_subject_list_train = np.array([])
    
    if all_train_indices:
        clips_tr_aligned = clips_tr.loc[all_train_indices]
        y_train = clips_tr_aligned["label"].values
        
        if 'patient' in clips_tr_aligned.index.names:
            sample_subject_list_train = clips_tr_aligned.index.get_level_values('patient').to_numpy()
        elif 'patient' in clips_tr_aligned.columns:
            sample_subject_list_train = clips_tr_aligned['patient'].to_numpy()
    
    print(f"✅ Training complete: {X_train.shape if X_train.size > 0 else 'Empty'}")
    
    # Process test data
    print(f"\n⏳ Processing Test Set...")
    
    test_session_groups = list(clips_te.groupby(level=["patient", "session"]))
    print(f"🚀 Processing {len(test_session_groups)} test sessions...")
    
    test_results = Parallel(n_jobs=max_workers, verbose=1)(
        delayed(process_session)(
            session_group,
            DATA_ROOT / "test",
            bp_filter_coeffs,
            notch_filter_coeffs,
            F_S
        ) for session_group in tqdm(test_session_groups, desc="Test Sessions")
    )
    
    # Collect test results
    X_test_list = []
    
    for features_list, indices_list in test_results:
        X_test_list.extend(features_list)
    
    X_test = np.array(X_test_list) if X_test_list else np.array([])
    
    print(f"✅ Test complete: {X_test.shape if X_test.size > 0 else 'Empty'}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    if X_train.size > 0:
        np.save(LOCAL_DATA_ROOT / "extracted_features/X_train_simplified.npy", X_train)
        print(f"   ✅ Saved X_train: {X_train.shape}")
    
    if y_train.size > 0:
        np.save(LOCAL_DATA_ROOT / "labels/y_train_simplified.npy", y_train)
        print(f"   ✅ Saved y_train: {y_train.shape}")
    
    if sample_subject_list_train.size > 0:
        np.save(LOCAL_DATA_ROOT / "extracted_features/sample_subject_array_train_simplified.npy", 
               sample_subject_list_train)
        print(f"   ✅ Saved subjects: {sample_subject_list_train.shape}")
    
    if X_test.size > 0:
        np.save(LOCAL_DATA_ROOT / "extracted_features/X_test_simplified.npy", X_test)
        print(f"   ✅ Saved X_test: {X_test.shape}")
    
    print(f"\n🎉 Feature extraction complete!")
    print(f"⏰ Total time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print feature statistics
    if X_train.size > 0:
        print(f"\n📊 Feature Statistics:")
        print(f"   Features per sample: {X_train.shape[1]}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0] if X_test.size > 0 else 0}")
        print(f"   Feature range: [{np.min(X_train):.4f}, {np.max(X_train):.4f}]")
        print(f"   Non-zero features: {np.count_nonzero(X_train)} / {X_train.size}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified EEG Feature Extraction')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--test-mode', action='store_true', help='Process limited data for testing')
    parser.add_argument('--max-workers', type=int, default=None, help='Number of parallel workers')
    args = parser.parse_args()
    main(verbose=args.verbose, test_mode=args.test_mode, max_workers=args.max_workers)
