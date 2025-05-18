import numpy as np
from scipy import signal

def design_bandpass_filter(lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Designs a Butterworth bandpass filter (SOS format).

    Args:
        lowcut (float): Lower cutoff frequency in Hz.
        highcut (float): Upper cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the Butterworth filter.

    Returns:
        np.ndarray: SOS filter coefficients.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Adjust frequencies to be strictly within (0, 1) normalized range
    if high >= 1.0:
        high = 0.99999 # Slightly less than 1 to avoid edge issues
    if low <= 0:
        low = 1e-7    # Very small positive number
    
    if low >= high:
        raise ValueError(f"Lowcut frequency ({lowcut} Hz) must be lower than highcut frequency ({highcut} Hz) after normalization.")
        
    sos = signal.butter(order, [low, high], btype='bandpass', analog=False, output='sos', fs=None)
    return sos

def design_notch_filter(notch_freq: float, quality_factor: float, fs: float) -> np.ndarray:
    """
    Designs a notch filter (IIR) and returns SOS coefficients.

    Args:
        notch_freq (float): Frequency to remove in Hz (e.g., 60 Hz for powerline).
        quality_factor (float): Quality factor Q. Higher Q means a narrower notch.
        fs (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: SOS filter coefficients.
    """
    if notch_freq <= 0 or notch_freq >= fs / 2:
        raise ValueError(f"Notch frequency ({notch_freq} Hz) must be positive and less than Nyquist frequency ({fs/2} Hz).")

    b, a = signal.iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    sos = signal.tf2sos(b, a)
    return sos

def apply_sos_filter(data: np.ndarray, sos_coeffs: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Applies a pre-designed SOS filter to data using zero-phase filtering.

    Args:
        data (np.ndarray): Input signal. Expected (time_points, channels) or (time_points,).
        sos_coeffs (np.ndarray): SOS filter coefficients.
        axis (int): Axis along which to filter. Default 0 for (time, channels) data.

    Returns:
        np.ndarray: Filtered signal data.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    return signal.sosfiltfilt(sos_coeffs, data, axis=axis).copy()

def apply_combined_time_domain_filters(
    data: np.ndarray,
    fs: float,
    bandpass_freqs: tuple = (0.5, 50.0),
    bandpass_order: int = 4,
    apply_notch: bool = True,
    notch_freq: float = 60.0,
    notch_quality_factor: float = 30.0,
    filter_order: str = 'bandpass_first'
) -> np.ndarray:
    """
    Applies bandpass and optionally a notch filter in the time domain.

    Args:
        data (np.ndarray): Input signal (time_points, channels).
        fs (float): Sampling frequency.
        bandpass_freqs (tuple): (lowcut, highcut) for bandpass filter.
        bandpass_order (int): Order for Butterworth bandpass.
        apply_notch (bool): Whether to apply the notch filter.
        notch_freq (float): Frequency for the notch filter.
        notch_quality_factor (float): Q factor for the notch filter.
        filter_order (str): 'bandpass_first' or 'notch_first'.

    Returns:
        np.ndarray: Filtered signal.
    """
    filtered_data = data.copy()

    bp_sos = design_bandpass_filter(bandpass_freqs[0], bandpass_freqs[1], fs, bandpass_order)
    
    if filter_order == 'bandpass_first':
        filtered_data = apply_sos_filter(filtered_data, bp_sos, axis=0)
        if apply_notch:
            notch_sos = design_notch_filter(notch_freq, notch_quality_factor, fs)
            filtered_data = apply_sos_filter(filtered_data, notch_sos, axis=0)
    elif filter_order == 'notch_first':
        if apply_notch:
            notch_sos = design_notch_filter(notch_freq, notch_quality_factor, fs)
            filtered_data = apply_sos_filter(filtered_data, notch_sos, axis=0)
        filtered_data = apply_sos_filter(filtered_data, bp_sos, axis=0)
    else:
        raise ValueError("filter_order must be 'bandpass_first' or 'notch_first'")
        
    return filtered_data

def filter_signal_fft(
    data: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    axis: int = 0 # Assuming axis 0 is the time axis for (time, channels)
) -> np.ndarray:
    """
    Filters a signal in the frequency domain using FFT and returns a time-domain signal.
    Zeros out frequency components outside the [lowcut, highcut] band.
    Assumes data is (time_points, channels) if 2D, and filtering is along time axis.

    Args:
        data (np.ndarray): Input signal data.
        fs (float): Sampling frequency.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        axis (int): Axis along which to perform FFT (time axis).

    Returns:
        np.ndarray: Filtered time-domain signal.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
    
    fft_data = np.fft.fft(data, axis=axis)
    freqs = np.fft.fftfreq(data.shape[axis], d=1/fs)
    fft_data_filtered = fft_data.copy()

    mask = (np.abs(freqs) < lowcut) | (np.abs(freqs) > highcut)
    
    if data.ndim == 1:
        fft_data_filtered[mask] = 0
    elif data.ndim == 2 and axis == 0:
        fft_data_filtered[mask, :] = 0
    # Removed axis=1 case for simplification, assuming (time, channels) and axis=0
    else:
        raise ValueError("Data must be 1D, or 2D with filtering along axis 0.")

    filtered_signal = np.fft.ifft(fft_data_filtered, axis=axis)
    return np.real(filtered_signal).copy()

def extract_log_band_spectrum_features(
    data: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 30.0,
    epsilon: float = 1e-8,
    axis: int = 0
) -> np.ndarray:
    """
    Computes FFT, takes the log-magnitude, and returns a slice of the spectrum.
    This is typically used for feature extraction.

    Args:
        data (np.ndarray): Input signal (time_points, channels).
        fs (float): Sampling frequency.
        lowcut (float): Lower frequency bound for the returned spectrum slice.
        highcut (float): Upper frequency bound.
        epsilon (float): Small value to avoid log(0).
        axis (int): Axis along which to compute FFT.

    Returns:
        np.ndarray: Slice of the log-magnitude spectrum.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a NumPy array.")
        
    fft_magnitudes = np.abs(np.fft.fft(data, axis=axis))
    log_spectrum = np.log(np.where(fft_magnitudes > epsilon, fft_magnitudes, epsilon))

    win_len = data.shape[axis]
    
    idx_low = int(np.ceil(lowcut * win_len / fs))
    idx_high = int(np.floor(highcut * win_len / fs))

    max_idx = win_len // 2 
    idx_low = min(max(0, idx_low), max_idx)
    idx_high = min(max(0, idx_high), max_idx)

    if idx_low >= idx_high:
        # Return an empty array of appropriate shape if the band is invalid
        # For 2D input (time, channels), output shape is (0, num_channels)
        num_channels_dim = data.shape[1] if data.ndim == 2 else 0
        if data.ndim == 1:
             return np.array([])
        elif data.ndim == 2 and axis == 0:
             return np.empty((0, num_channels_dim))
        # Fallback for other unhandled cases, though less likely with simplified input assumptions
        return np.array([]) 
        
    return log_spectrum[idx_low:idx_high]
