import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_clip(signal_data, title_prefix, sampling_rate, channel_names, offset_factor=10):
    """Plots an EEG clip with channels offset vertically and labels aligned
       to the vertical center of each trace."""

    if signal_data is None or not isinstance(signal_data, np.ndarray) or signal_data.ndim != 2:
        print(f"Invalid data provided for {title_prefix}. Skipping plot.")
        return

    actual_time_points, actual_num_channels = signal_data.shape
    print(
        f"Plotting {title_prefix} with shape: {(actual_time_points, actual_num_channels)}")

    # --- Create time vector based on actual data length ---
    time_vector = np.arange(actual_time_points) / sampling_rate

    plt.figure(figsize=(10, 8), dpi=100)

    # --- Robust Vertical Offset Calculation ---
    signal_std = np.nanstd(signal_data)
    min_std_dev = 1e-6
    if signal_std < min_std_dev:
        print(
            f"Warning: Signal standard deviation is very small ({signal_std:.2e}). Using default offset scale.")
        offset_scale = (np.nanmean(np.abs(signal_data)) + 1.0) * offset_factor
    else:
        offset_scale = signal_std * offset_factor
    if offset_scale == 0:
        offset_scale = offset_factor  # Fallback

    # 'offsets' array determines the vertical baseline shift for each channel's plot
    offsets = np.arange(actual_num_channels) * offset_scale

    # --- Apply offsets to data ---
    offsetted_data = signal_data + offsets

    # --- Check dimensions before plotting ---
    if time_vector.shape[0] == signal_data.shape[0]:
        # --- Plotting Signals ---
        # Plot the data *after* offsets have been added
        plt.plot(time_vector, offsetted_data)

        # --- Setting Y-Ticks and Labels (Centered on Trace Mean) ---
        # Calculate the mean vertical position of each trace
        channel_means = np.nanmean(offsetted_data, axis=0)

        # Use the calculated mean positions for the y-ticks.
        # This aligns the label with the average value of the signal trace.
        plt.yticks(channel_means, channel_names[:actual_num_channels])

        plt.xlabel('Time [s]')
        plt.ylabel('Channel')
        plt.title(f'{title_prefix} ({actual_time_points} points)')
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.grid(True, axis='x', linestyle='-', alpha=0.7)

        # plt.tight_layout(pad=1.5) # Commented out: can sometimes interfere
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95,
                            bottom=0.05)  # Manual adjustment

        plt.show()
    else:
        print(
            f"Error: Mismatch plotting {title_prefix}. Time vector shape {time_vector.shape} vs Data shape {signal_data.shape}")

def plot_training_loss(train_losses, num_epochs):
    """Plots the training loss over epochs."""
    if train_losses is None or not isinstance(train_losses, list) or len(train_losses) != num_epochs:
        print(f"Invalid training loss data. Expected {num_epochs} epochs.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, num_epochs + 1), train_losses, marker="o")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()