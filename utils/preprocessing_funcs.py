# Relative import (importing modules within the same package)

from .imports import (
    plt,
    welch,
    signal,
    np,
    pd,
    StratifiedKFold,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    clone,
    RandomOverSampler,
    RobustScaler,
    tqdm,
)


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


def spectral_entropy(psd):
    """Spectral entropy of a power‐spectral density array."""
    psd_sum = psd.sum()
    p_norm = psd / (psd_sum + 1e-12)  # avoid /0
    return -(p_norm * np.log2(p_norm + 1e-12)).sum()


# --------------------------- Feature extractor ----------------------------------------------#
def extract_features(signal, session, start, end, fs=250):
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

    # convert to NumPy & NaN guard --------------------------------------
    features = np.asarray(features, dtype=float)
    if np.any(np.isnan(features)):
        print("NaN detected in extracted features.")
        print(f"Session: {session} | Start: {start} | End: {end}")
        features = np.nan_to_num(features, nan=0.0)

    return features


# ---------------------- Function to process one session to extract features -------------------------------#


def process_session(
    session,
    signal_path,
    bp_filter,
    notch_filter,
    segment_len=250 * 12,
    f_s=250,
):
    features_list = []

    # Load signal and extract the window
    signal_session = pd.read_parquet(signal_path)
    session_t0 = int(session.iloc[0]["start_time"] * f_s)
    session_tf = int(session.iloc[-1]["end_time"] * f_s)
    session_signal = signal_session.iloc[session_t0:session_tf].values

    # Filter
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
    for i, start in enumerate(range(0, len(session_signal), segment_len)):
        end = start + segment_len
        segment = session_signal[start:end]
        features = extract_features(segment, session, start, end)
        features_list.append(features)

    return features_list


# --------------------- Function to evaluate performance basic ML models ------------------------#


# MODEL EVALAUTION FUNCTION
def model_evaluation(
    model,
    X,
    y,
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

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # Use StratifiedKFold to preserve class proportions in each fold

    for train_idx, val_idx in tqdm(
        kf.split(X, y), total=kf.get_n_splits(), desc="CV folds"
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
