import os
import sys
from pathlib import Path
from joblib import Parallel, delayed  # For parallel processing
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import welch, butter, iirnotch, sosfiltfilt, tf2sos

from src.utils.signal import spectral_entropy
from src.utils.dataset import ensure_eeg_multiindex

def _extract_channel_features(channel_signal: np.ndarray, fs: int,
                              bands: Dict[str, Tuple[float, float]]) -> List[float]:
    rms_val = np.sqrt(np.mean(channel_signal**2))
    line_len = np.sum(np.abs(np.diff(channel_signal)))
    dx = np.diff(channel_signal)
    var_x = np.var(channel_signal)
    hj_mob, hj_cmp = 0.0, 0.0
    if var_x > 1e-10:
        var_dx = np.var(dx)
        hj_mob = np.sqrt(var_dx / (var_x + 1e-12))
        if var_dx > 1e-10:
            ddx = np.diff(dx)
            hj_cmp = np.sqrt(np.var(ddx) / (var_dx + 1e-12))

    nperseg_welch = min(len(channel_signal), 2 * fs)
    num_spectral_related_features = 7
    spec_ent_val = 0.0
    bp_values = {name: 0.0 for name in bands}
    rel_alpha_val, rel_theta_val, theta_over_alpha_val = 0.0, 0.0, 0.0

    if len(channel_signal) >= 2 and nperseg_welch >= 2:
        try:
            freqs, psd = welch(channel_signal, fs=fs, nperseg=nperseg_welch)
            total_pow = np.sum(psd) + 1e-12
            for name, (lo, hi) in bands.items():
                idx = (freqs >= lo) & (freqs < hi)
                bp_values[name] = np.trapezoid(
                    psd[idx], freqs[idx]) if np.any(idx) else 0.0

            if total_pow > 1e-10:
                rel_alpha_val = bp_values.get("alpha", 0.0) / total_pow
                rel_theta_val = bp_values.get("theta", 0.0) / total_pow
                spec_ent_val = spectral_entropy(psd)
            if bp_values.get("alpha", 0.0) > 1e-12:
                theta_over_alpha_val = bp_values.get(
                    "theta", 0.0) / bp_values.get("alpha", 0.0)
        except ValueError:
            pass
    return [
        rms_val, line_len, hj_mob, hj_cmp, spec_ent_val,
        bp_values.get("alpha", 0.0), bp_values.get("beta", 0.0), bp_values.get(
            "theta", 0.0), bp_values.get("gamma", 0.0),
        rel_alpha_val, rel_theta_val, theta_over_alpha_val
    ]


def _extract_segment_features(segment_signal: np.ndarray, fs: int = 250) -> np.ndarray:
    all_channel_features: List[float] = []
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (
        8, 13), "beta": (13, 30), "gamma": (30, 50)}
    num_channels = segment_signal.shape[1]
    for ch_idx in range(num_channels):
        ch_feats = _extract_channel_features(
            segment_signal[:, ch_idx], fs, bands)
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
    num_features_per_channel: int = 12
) -> Tuple[List[np.ndarray], List[Any]]:
    group_name, session_df = session_group

    # Calculate expected feature size
    expected_feature_size = num_channels * num_features_per_channel
    empty_features = np.zeros(expected_feature_size, dtype=float)

    features_list_for_session: List[np.ndarray] = []
    all_original_indices: List[Any] = []

    if 'signals_path' not in session_df.columns or session_df.empty:
        # Return empty features for all rows in this session
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        return features_list_for_session, all_original_indices

    relative_signal_file = session_df['signals_path'].iloc[0]
    full_session_signal_path = base_signal_path / relative_signal_file

    if not full_session_signal_path.exists():
        # Return empty features for all rows in this session
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        return features_list_for_session, all_original_indices

    try:
        session_signal_df = pd.read_parquet(full_session_signal_path)
        session_signal_values = session_signal_df.values
    except Exception as e:
        print(
            f"Error loading session signal {full_session_signal_path} for group {group_name}: {e}. Using empty features for all segments.")
        # Return empty features for all rows in this session
        for original_idx, _ in session_df.iterrows():
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
        return features_list_for_session, all_original_indices

    processed_signal = sosfiltfilt(
        bp_filter_coeffs, session_signal_values, axis=0)
    if notch_filter_coeffs is not None:
        processed_signal = sosfiltfilt(
            notch_filter_coeffs, processed_signal, axis=0)

    avg_reference = np.mean(processed_signal, axis=1, keepdims=True)
    processed_signal = processed_signal - avg_reference
    mean_per_channel = np.mean(processed_signal, axis=0, keepdims=True)
    std_per_channel = np.std(processed_signal, axis=0, keepdims=True)
    processed_signal = (processed_signal - mean_per_channel) / \
        (std_per_channel + 1e-6)

    for original_idx, row in session_df.iterrows():
        try:
            t0 = int(row["start_time"] * f_s)
            tf = int(row["end_time"] * f_s)
            max_len = processed_signal.shape[0]
            segment_valid = True

            if t0 < 0 or t0 >= max_len:
                segment_valid = False
            elif tf > max_len:
                tf = max_len
                if t0 >= tf:
                    segment_valid = False
            elif t0 >= tf:
                segment_valid = False

            if not segment_valid:
                features_list_for_session.append(empty_features.copy())
                all_original_indices.append(original_idx)
                continue

            segment = processed_signal[t0:tf, :]
            if segment.shape[0] < 2:
                features_list_for_session.append(empty_features.copy())
                all_original_indices.append(original_idx)
                continue

            features = _extract_segment_features(segment, fs=f_s)
            features_list_for_session.append(features)
            all_original_indices.append(original_idx)
        except Exception:
            features_list_for_session.append(empty_features.copy())
            all_original_indices.append(original_idx)
            continue
    return features_list_for_session, all_original_indices


IS_SCITAS = False
if __name__ == "__main__":
    LOCAL_DATA_ROOT = Path("data")
    (LOCAL_DATA_ROOT / "extracted_features").mkdir(parents=True, exist_ok=True)
    (LOCAL_DATA_ROOT / "labels").mkdir(parents=True, exist_ok=True)
    
    DATA_ROOT= Path("/home/ogut/data") if IS_SCITAS else Path("data")
    print(f"Looking for data in: {LOCAL_DATA_ROOT.resolve()}")

    try:
        clips_tr_full = pd.read_parquet(
            DATA_ROOT / "train" / "segments.parquet")
        clips_te_full = pd.read_parquet(
            DATA_ROOT / "test" / "segments.parquet")
    except FileNotFoundError as e:
        print(f"Error: Parquet file not found. Details: {e}")
        sys.exit(1)
    print(clips_tr_full.shape)

    # --- Ensure MultiIndex ---
    clips_tr_full = ensure_eeg_multiindex(clips_tr_full, id_col_name='id')
    clips_te_full = ensure_eeg_multiindex(clips_te_full, id_col_name='id')

    # Filter out rows with NaN labels in training set
    print("Filtering training set for valid labels...")
    clips_tr_for_labels = clips_tr_full[~clips_tr_full.label.isna()].copy()
    clips_te = clips_te_full.copy()

    F_S = 250
    BP_FILTER_FREQS = (0.5, 50.0)
    NOTCH_FREQ_HZ = 60.0
    NOTCH_Q_FACTOR = 30.0

    print(
        f"\nDesigning filters: BP={BP_FILTER_FREQS}Hz, Notch={NOTCH_FREQ_HZ}Hz, FS={F_S}Hz")
    bp_filter_coeffs_sos = butter(
        4, BP_FILTER_FREQS, btype="bandpass", output="sos", fs=F_S)
    notch_filter_coeffs_ba = iirnotch(
        w0=NOTCH_FREQ_HZ, Q=NOTCH_Q_FACTOR, fs=F_S)
    notch_filter_coeffs_sos = tf2sos(*notch_filter_coeffs_ba)

    # --- Training Set Feature Extraction ---
    print("\n⏳ Processing Training Set...")

    train_session_groups = []
    try:
        # Prefer grouping by index levels if they exist as per user's description
        if all(level in clips_tr_for_labels.index.names for level in ["patient", "session"]):
            print(
                "   Grouping clips_tr_for_labels by index levels: ['patient', 'session']")
            train_session_groups = list(
                clips_tr_for_labels.groupby(level=["patient", "session"]))
        # Fallback to columns if index levels are not set as expected
        elif all(col in clips_tr_for_labels.columns for col in ["patient", "session"]):
            print(
                "   Grouping clips_tr_for_labels by columns: ['patient', 'session']")
            train_session_groups = list(
                clips_tr_for_labels.groupby(["patient", "session"]))
        else:
            raise ValueError(
                "Could not find 'patient' and 'session' as index levels or columns in clips_tr_for_labels for grouping.")
    except Exception as e:
        print(f"Error during groupby for training set: {e}")
        print(f"clips_tr_for_labels index: {clips_tr_for_labels.index.names}")
        print(f"clips_tr_for_labels columns: {clips_tr_for_labels.columns}")
        sys.exit(1)

    train_processing_results = Parallel(n_jobs=-1)(
        delayed(process_session_for_features)(
            session_group,
            base_signal_path=LOCAL_DATA_ROOT / "train",
            bp_filter_coeffs=bp_filter_coeffs_sos,
            notch_filter_coeffs=notch_filter_coeffs_sos,
            f_s=F_S
        )
        for session_group in tqdm(train_session_groups, desc="Processing Train Sessions")
    )

    X_train_list = []
    all_train_indices = []
    for session_feat_list, session_indices_list in train_processing_results:
        if session_feat_list:
            X_train_list.extend(session_feat_list)
            all_train_indices.extend(session_indices_list)

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
        print("Warning: No training features were extracted. y_train and sample_subject_list_train will be empty.")

    # --- Test Set Feature Extraction ---
    print("\n⏳ Processing Test Set...")
    test_session_groups = []
    try:
        if all(level in clips_te.index.names for level in ["patient", "session"]):
            print(
                "   Grouping clips_te by index levels: ['patient', 'session']")
            test_session_groups = list(
                clips_te.groupby(level=["patient", "session"]))
        elif all(col in clips_te.columns for col in ["patient", "session"]):
            print("   Grouping clips_te by columns: ['patient', 'session']")
            test_session_groups = list(
                clips_te.groupby(["patient", "session"]))
        else:
            # If test set doesn't have patient/session, how should it be grouped for signals_path?
            # This might require a different strategy if signals_path is not per-segment.
            # For now, assume it also has a structure that allows similar grouping.
            raise ValueError(
                "Could not find 'patient' and 'session' as index levels or columns in clips_te for grouping.")
    except Exception as e:
        print(f"Error during groupby for test set: {e}")
        print(f"clips_te index: {clips_te.index.names}")
        print(f"clips_te columns: {clips_te.columns}")
        sys.exit(1)

    test_processing_results = Parallel(n_jobs=-1)(
        delayed(process_session_for_features)(
            session_group,
            base_signal_path=LOCAL_DATA_ROOT / "test",
            bp_filter_coeffs=bp_filter_coeffs_sos,
            notch_filter_coeffs=notch_filter_coeffs_sos,
            f_s=F_S
        )
        for session_group in tqdm(test_session_groups, desc="Processing Test Sessions")
    )

    X_test_list = []
    # For test set, we collect all features (including empty ones for failed processing)
    for session_feat_list, _ in test_processing_results:
        if session_feat_list:
            X_test_list.extend(session_feat_list)
    X_test = np.array(X_test_list) if X_test_list else np.array([])

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
