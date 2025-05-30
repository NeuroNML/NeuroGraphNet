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
