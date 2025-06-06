# ------------------------- Imports ---------------------------#

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy
from scipy.linalg import toeplitz
from scipy.signal import stft
import pywt


from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone
from imblearn.over_sampling import RandomOverSampler


# ---------------------------------- Frequency visualization ----------------------------------#
def visualize_frequencies(segment, channels, f_s=250):
    # Set up subplots: 4 rows x 5 columns
    fig, axes = plt.subplots(4, 5, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    # Apply filter

    # Loop through all channels
    for i, channel in enumerate(channels):
        ch_signal = segment[channel].values
        f, Pxx = welch(
            ch_signal, fs=f_s, nperseg=f_s * 2
        )  # 2s window; f: frequencies and corresponding power

        ax = axes[i]
        ax.semilogy(
            f, Pxx
        )  # Power spectra plotted with log due to significant magnitude differences
        ax.set_title(channel)
        ax.set_xlim(0, max(f))
        ax.grid(True)

    # Global figure settings
    axes[19].axis("off")  # Eliminate last axis
    fig.suptitle("Power Spectral Density - All Channels", fontsize=16)
    plt.tight_layout()
    plt.show()


# ---------------------------- Time filtering -------------------------------------------------------#
def time_filtering(x, bp_filter, notch_filter):
    """Apply bandpass + notch filtering to EEG signal in the time domain
    x, output: (time, channels)
    """

    x_filt = signal.sosfiltfilt(bp_filter, x, axis=0)
    x_filt = signal.sosfiltfilt(notch_filter, x_filt, axis=0)

    return x_filt.copy()


def spectral_entropy(psd): # Shannon entropy
    """Spectral entropy of a power‐spectral density array."""
    psd_sum = psd.sum()
    p_norm = psd / (psd_sum + 1e-12)  # avoid /0
    return -(p_norm * np.log2(p_norm + 1e-12)).sum()


# --------------------------- Feature extractor ----------------------------------------------#
def extract_de_features(signal, fs=250):
    """
    Extract Differential Entropy (DE) features per band per channel.

    Returns:
    - 1D feature vector: (n_channels × n_bands) DE features
    """
    features = []
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        freqs, psd = welch(x, fs=fs, nperseg=2 * fs)

        for name, (lo, hi) in bands.items():
            idx = (freqs >= lo) & (freqs < hi)
            band_psd = psd[idx]

            # Compute band power (integral of PSD over band)
            band_power = np.trapz(band_psd, freqs[idx])

            # DE assumes Gaussian: DE = 0.5 * log(2πeσ²)
            # σ² is approximated from band power
            sigma_sq = band_power
            de = 0.5 * np.log(2 * np.pi * np.e * (sigma_sq + 1e-12))

            features.append(de)

    return np.asarray(features, dtype=float)


def extract_features(signal, fs=250):
    """
    Extract EEG features.
    signal : ndarray (time, channels), already re-referenced
    Returns : 1-D feature vector (12 features × n_channels)
    """
    features = []
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    for ch in range(signal.shape[1]):
        x = signal[:, ch]

        # Time domain
        rms_val = np.sqrt(np.mean(x**2))
        line_len = np.sum(np.abs(np.diff(x)))

        # Hjorth parameters
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x = np.var(x)
        hj_mob = np.sqrt(np.var(dx) / (var_x + 1e-12))
        hj_cmp = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12))

        # ------------- Frequency domain ----------------#
        freqs, psd = welch(x, fs=fs, nperseg=2 * fs)
        total_pow = psd.sum()

        # Absolute band powers
        bp = {}
        for name, (lo, hi) in bands.items():
            idx = (freqs >= lo) & (freqs < hi)
            bp[name] = np.trapz(psd[idx], freqs[idx])

        # Relative powers
        rel_alpha = bp["alpha"] / (total_pow + 1e-12)
        rel_theta = bp["theta"] / (total_pow + 1e-12)

        # Band ratio
        theta_over_alpha = bp["theta"] / (bp["alpha"] + 1e-12)

        # Spectral entropy
        spec_ent = spectral_entropy(psd)

        # ---------- collect 12 features for this channel ---------------
        ch_feats = [
            rms_val,
            line_len,
            hj_mob,
            hj_cmp,
            spec_ent,
            bp["alpha"],
            bp["beta"],
            bp["theta"],
            bp["gamma"],
            rel_alpha,
            rel_theta,
            theta_over_alpha,
        ]
        features.extend(ch_feats)

    # Convert to np and create guardarail for NaN
    features = np.asarray(features, dtype=float)
    """
    if np.any(np.isnan(features)):
        print("NaN detected in extracted features.")
        print(f"Session: {session} | Start: {start} | End: {end}")
        features = np.nan_to_num(features, nan=0.0)
    """

    return features


# ----------------------------- Function to extract features per window --------------------------#

def bandpower(psd, freqs, band):
    mask = (freqs >= band[0]) & (freqs < band[1])
    return np.trapz(psd[mask], freqs[mask])


def extract_power_features_window(signal, fs=250):
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "gamma": (30, 50),
    }

    n_channels = signal.shape[0]
    features = []

    for ch in range(n_channels):
        x = signal[ch]

        # Welch PSD
        f, psd = welch(x, fs=fs, nperseg=len(x))

        # Band powers
        theta_pow = bandpower(psd, f, bands["theta"])
        alpha_pow = bandpower(psd, f, bands["alpha"])
        gamma_pow = bandpower(psd, f, bands["gamma"])

        # Spectral entropy
        spec_ent = spectral_entropy(psd)

        ch_feats = [theta_pow, alpha_pow, gamma_pow, spec_ent]
        features.extend(ch_feats)

    return np.array(features, dtype=float)


def compute_bispectrum(x, fs, nfft=250):
    f, t, Zxx = stft(x, fs=fs, nperseg=nfft)
    S = Zxx.mean(axis=1)
    S_conj = np.conj(S)
    B = np.zeros((len(f), len(f)), dtype=complex)
    for i in range(len(f)):
        for j in range(len(f)):
            k = i + j
            if k < len(f):
                B[i, j] = S[i] * S[j] * S_conj[k]
    return B, f

def weighted_mean(arr):
    weights = np.arange(1, len(arr) + 1)
    return np.average(arr, weights=weights)

def weighted_variance(arr):
    weights = np.arange(1, len(arr) + 1)
    mean = np.average(arr, weights=weights)
    return np.average((arr - mean) ** 2, weights=weights)

def extract_bispectrum_window(signal, fs=250):
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "delta": (0.5, 4),
        "beta":  (13, 30),
        "gamma": (30, 50),
    }

    n_channels = signal.shape[1]
    features = []

    for ch in range(n_channels):
        x = signal[:, ch]
        B, f_bi = compute_bispectrum(x, fs=fs)
        B = np.abs(B)
        ch_feats = []

        for band, (lo, hi) in bands.items():
            band_mask = (f_bi >= lo) & (f_bi < hi)
            B_band = B[np.ix_(band_mask, band_mask)]
            diag = np.diag(B_band)

            log_diag_energy = np.sum(np.log(diag + 1e-12))
            log_energy = np.sum(np.log(B_band + 1e-12))

            if band == "theta":
                ch_feats.extend([log_diag_energy, log_energy])
            elif band == "delta":
                ch_feats.extend([log_energy, log_diag_energy])
            elif band == "alpha":
                entropy_ = entropy(diag / (np.sum(diag) + 1e-12))
                ch_feats.extend([log_diag_energy, log_energy, entropy_])
            elif band == "beta":
                ch_feats.append(log_energy)
            elif band == "gamma":
                diag_wvar = weighted_variance(diag)
                diag_wmean = weighted_mean(diag)
                entropy_ = entropy(diag / (np.sum(diag) + 1e-12))
                quad_entropy = -np.sum((diag**2 / (np.sum(diag**2) + 1e-12)) * np.log(diag**2 / (np.sum(diag**2) + 1e-12) + 1e-12))
                ch_feats.extend([log_energy, log_diag_energy, diag_wvar, diag_wmean, entropy_, quad_entropy])

        features.extend(ch_feats)

    return np.array(features, dtype=float)


def extract_de_window_features(signal, fs=250):
    """
    Extract Differential Entropy (DE) features per band per channel.

    Returns:
    - 1D feature vector: (n_channels × n_bands) DE features
    """
    features = []
    bands = {
 
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    for ch in range(signal.shape[1]):
        x = signal[:, ch]
        freqs, psd = welch(x, fs=fs, nperseg=2 * fs)

        for name, (lo, hi) in bands.items():
            idx = (freqs >= lo) & (freqs < hi)
            band_psd = psd[idx]

            # Compute band power (integral of PSD over band)
            band_power = np.trapz(band_psd, freqs[idx])

            # DE assumes Gaussian: DE = 0.5 * log(2πeσ²)
            # σ² is approximated from band power
            sigma_sq = band_power
            de = 0.5 * np.log(2 * np.pi * np.e * (sigma_sq + 1e-12))

            features.append(de)


    return np.asarray(features, dtype=float)


def extract_wavelet_window_features(signal, fs=250):
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "gamma": (30, 50),
    }

    n_channels = signal.shape[1]
    features = []

    for ch in range(n_channels):
        x = signal[:, ch]
        coeffs, freqs = pywt.cwt(
            x, scales=np.arange(1, 128), wavelet="morl", sampling_period=1 / fs
        )
        power = np.abs(coeffs) ** 2
        ch_feats = []

        for band, (lo, hi) in bands.items():
            if band not in ["theta", "delta", "alpha", "gamma"]:
                continue

            band_mask = (freqs >= lo) & (freqs < hi)
            band_power = power[band_mask, :].mean(axis=0)

            if band == "theta":
                ch_feats.extend([skew(band_power), kurtosis(band_power)])
            elif band == "delta":
                ch_feats.extend([
                    skew(band_power), kurtosis(band_power),
                    np.std(band_power), np.mean(band_power)
                ])
            elif band == "alpha":
                ch_feats.extend([
                    skew(band_power), kurtosis(band_power),
                    np.mean(band_power),
                    entropy(np.abs(band_power) / (np.sum(np.abs(band_power)) + 1e-12))
                ])
            elif band == "gamma":
                ch_feats.extend([
                    skew(band_power), np.std(band_power)
                ])

        features.extend(ch_feats)

    return np.array(features, dtype=float)

def extract_window_features(signal, fs=250):
    bispec_feats = extract_bispectrum_window(signal, fs)
    de_feats = extract_de_window_features(signal, fs)
    wavelet_feats = extract_wavelet_window_features(signal, fs)
    power_feats =  extract_power_features_window(signal, fs)
    linelen =  np.sum(np.abs(np.diff(signal, axis=1)), axis=1, keepdims=True) # Linelength per channel

    all_feats = np.concatenate([bispec_feats, de_feats, wavelet_feats, power_feats, linelen], axis=1)
    return all_feats  # shape: (n_channels, total_features)


def extract_features_over_windows(segment, fs=250, window_size=1.0, overlap=0.5):
    """
    eeg_segment: np.ndarray of shape (n_channels, total_samples)
    Returns: np.ndarray of shape (n_windows, n_features_per_window)
    """
    window_len = int(window_size * fs)
    overlap_len = int(overlap * fs)
    step = window_len - overlap_len
    n_channels, total_len = segment.shape

    features_per_window = []

    for start in range(0, total_len - window_len + 1, step):
        window = segment[:, start:start + window_len]
        feat_vec = extract_window_features(window, fs=fs)
        features_per_window.append(feat_vec)

    features_per_window = np.stack(features_per_window)  # shape: (n_windows, n_channels, n_features)
    features_per_window = np.transpose(features_per_window, (1, 0, 2))  # -> (n_channels, n_windows, n_features)
    return features_per_window




# ---------------------- Function to process one session to extract features -------------------------------#


def process_session(session_df, signal_path, bp_filter, notch_filter, f_s=250):
    features_list = []

    # Load signal and extract the window
    session_signal = pd.read_parquet(signal_path)  # (time, channels)

    # Filter - Input: (time, channels)
    session_signal = time_filtering(
        session_signal,
        bp_filter=bp_filter,
        notch_filter=notch_filter,
    )

    # Re-referencing: average reference across channels
    avg_reference = np.mean(session_signal, axis=1, keepdims=True)
    session_signal = session_signal - avg_reference

    # Normalize
    mean = np.mean(session_signal, axis=0, keepdims=True)
    std = np.std(session_signal, axis=0, keepdims=True)
    eps = 1e-6  # Small value to avoid division by zero
    session_signal = (session_signal - mean) / (std + eps)

    # Segment and extract
    for _, row in session_df.iterrows():
        t0 = int(row["start_time"] * f_s)
        tf = int(row["end_time"] * f_s)
        segment = session_signal[t0:tf]
        #features = extract_de_features(segment)
        features = extract_features_over_windows(segment) # 
        features_list.append(features)

    return features_list


# --------------------- Function to evaluate performance basic ML models ------------------------#


# MODEL EVALAUTION FUNCTION
def model_evaluation(
    model,
    X,
    y,
    sample_subject_array,
    n_features,
    n_splits=5,
    random_state=42,
    oversampler=True,
    show_matrix=False,
    return_importances=False,
    n_channels=19,
):
    """
    Returns (mean_f1, std_f1) for a model on (X, y) with optional over-sampling.
    Feature importance can also be returned if model has the capability
    """
    f1_scores, conf_matrices = [], []
    importances = []

    kf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    # Use Stratified to preserve class proportions in each fold; Group: so same patient is not in both training and validation

    for train_idx, val_idx in tqdm(
        kf.split(X, y, groups=sample_subject_array),
        total=kf.get_n_splits(),
        desc="CV folds",
    ):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standarize data according to train fold
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Oversample training fold
        if oversampler:
            sampler = RandomOverSampler(random_state=random_state)
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # Fit and  predict
        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        # Evaluate
        f1_scores.append(f1_score(y_val, y_pred, average="macro"))
        conf_matrices.append(confusion_matrix(y_val, y_pred))

        # Return importances
        if return_importances:
            imp = clf.feature_importances_
            importances.append(imp.reshape(n_channels, n_features))
            # Since features are ordered by electrode first -> can just reshape as a function of electrodes

    f1_scores = np.array(f1_scores)
    avg_cm = np.mean(conf_matrices, axis=0).astype(int)

    print(f" F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

    if return_importances:
        importances = np.stack(
            importances, axis=0
        )  # Joins arrays along new axis -> each fold new layer/(folds, channels, features)
        mean_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        return f1_scores.mean(), f1_scores.std(), mean_imp, std_imp

    if show_matrix:
        ConfusionMatrixDisplay(avg_cm, display_labels=np.unique(y)).plot(cmap="Blues")
        plt.title("Avg Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return f1_scores.mean(), f1_scores.std()



