if __name__ == "__main__":

    # --------------------- General imports --------------------- #

    import sys
    import os
    from pathlib import Path
    from joblib import Parallel, delayed

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    from scipy import signal

    # --------------------- Custom imports --------------------- #
    # Absolute paths - add utils directory to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils.preprocessing_funcs import process_session

    # --------------- Define path to extract clips -----------------------------------

    DATA_ROOT = Path(
        "data"
    )  # No need to go up one directory since added all dirs before

    clips_tr = pd.read_parquet(DATA_ROOT / "train/segments.parquet")
    clips_te = pd.read_parquet(DATA_ROOT / "test/segments.parquet")

    # Filter NaN values out of clips_tr
    clips_tr = clips_tr[~clips_tr.label.isna()]

    # -------------------------- Extract selected features --------------------------------------#

    # Define frequency filters
    f_s = 250  # sampling frequency

    # Bandpasss filter
    bp_filter = signal.butter(4, (0.5, 50), btype="bandpass", output="sos", fs=f_s)

    # Notch filter
    notch_filter = signal.iirnotch(w0=60, Q=30, fs=f_s)
    notch_filter = signal.tf2sos(
        *notch_filter
    )  # Need to convert to SOS format (same formar as bp_filter) for numerical stability

    # Training set

    X_train = []

    sessions = list(clips_tr.groupby(["patient", "session"]))
    sample_subject_list = clips_tr.reset_index().patient.values

    features = Parallel(n_jobs=-1)(
        delayed(process_session)(
            session,
            signal_path=DATA_ROOT / "train" / session["signals_path"].values[0],
            bp_filter=bp_filter,
            notch_filter=notch_filter,
        )
        for _, session in tqdm(sessions, desc="Processing sessions")
        if (DATA_ROOT / "train" / session["signals_path"].values[0])
        .resolve()
        .exists()  # Since have filtered out some signals from clips_tr
    )

    # features: list of (num_segments, features) (one session) -> need to eliminate list -> 2D array
    for session in features:
        X_train.extend(session)

    X_train = np.array(X_train)

    # Test set

    X_test = []

    sessions = list(clips_te.groupby(["patient", "session"]))

    features = Parallel(n_jobs=-1)(
        delayed(process_session)(
            session,
            signal_path=DATA_ROOT / "test" / session["signals_path"].values[0],
            bp_filter=bp_filter,
            notch_filter=notch_filter,
        )
        for _, session in tqdm(sessions, desc="Processing sessions")
    )

    for session in features:
        X_test.extend(session)

    X_test = np.array(X_test)

    # ------------------------------ Extract labels -------------------------------------------#
    y_train = clips_tr["label"].values

    # -------------------- Check correct values --------------------------------------#
    print("Final dataset shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)

    # ------------------------- Save arrays ---------------------------------------------#
    np.save(DATA_ROOT / "extracted_features/X_test_DE.npy", X_test)
    np.save(DATA_ROOT / "extracted_features/X_train_DE.npy", X_train)
    np.save(
        DATA_ROOT / "extracted_features/sample_subject_array.npy", sample_subject_list
    )

    np.save(DATA_ROOT / "labels/y_train.npy", y_train)
