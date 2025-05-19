from scipy import signal
import numpy as np

# bandpass filter for EEG signals: selects frequencies between 0.5 and 30Hz
SAMPLING_RATE=250
bp_filter = signal.butter(4, (0.5, 50), btype="bandpass", output="sos", fs=SAMPLING_RATE)
# notch filter to remove 60 Hz noise (fixed frequency)
notch_filter = signal.tf2sos(*signal.iirnotch(w0=60, Q=30, fs=SAMPLING_RATE))

def time_filtering(x):
    """Apply bandpass + notch filtering to EEG signal in the time domain
    x, output: (time, channels)
    """

    x_filt = signal.sosfiltfilt(bp_filter, x, axis=0)
    x_filt = signal.sosfiltfilt(notch_filter, x_filt, axis=0)

    return x_filt.copy()

def normalize(signal):
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        return (signal - mean) / (std + 1e-6)

def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep the frequencies between 0.5 and 30Hz"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]