import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_clip(signal_data, title_prefix, sampling_rate, channel_names, 
                  offset_factor=10, ax=None):
    """Plots an EEG clip on a given Matplotlib Axes or creates a new figure.
       Channels are offset vertically. Channel names are centered on the mean
       of their respective signal traces.
    """

    if signal_data.ndim != 2:
        error_msg = f"Warning: Invalid or empty signal_data provided for '{title_prefix}'. Shape: {getattr(signal_data, 'shape', type(signal_data))}."
        if ax is not None:
            ax.text(0.5, 0.5, error_msg, ha='center', va='center', wrap=True, color='red')
            ax.set_title(f"{title_prefix} - Data Error")
        else:
            print(error_msg)
        return
    
    if sampling_rate <= 0:
        error_msg = f"Warning: Invalid sampling_rate ({sampling_rate}) for '{title_prefix}'."
        if ax is not None:
            ax.text(0.5, 0.5, error_msg, ha='center', va='center', wrap=True, color='red')
            ax.set_title(f"{title_prefix} - Config Error")
        else:
            print(error_msg)
        return

    actual_time_points, actual_num_channels = signal_data.shape
    
    if actual_num_channels == 0 or actual_time_points == 0:
        error_msg = f"Warning: Signal data for '{title_prefix}' has zero channels or time points. Shape: {signal_data.shape}."
        if ax is not None:
            ax.text(0.5, 0.5, error_msg, ha='center', va='center', wrap=True, color='red')
            ax.set_title(f"{title_prefix} - Data Error")
        else:
            print(error_msg)
        return

    show_plot_later = False
    if ax is None:
        fig_width = 12
        fig_height = max(4, actual_num_channels * 0.4)
        fig, current_ax = plt.subplots(figsize=(fig_width, fig_height))
        ax = current_ax
        show_plot_later = True
    
    if len(channel_names) < actual_num_channels:
        generic_names = [f'Ch {i+len(channel_names)+1}' for i in range(actual_num_channels - len(channel_names))]
        effective_channel_names = list(channel_names) + generic_names
    else:
        effective_channel_names = channel_names[:actual_num_channels]

    time_vector = np.arange(actual_time_points) / sampling_rate

    channel_stds = np.nanstd(signal_data, axis=0)
    meaningful_stds = channel_stds[channel_stds > 1e-6]
    
    if len(meaningful_stds) > 0:
        typical_std = np.nanmean(meaningful_stds)
        offset_scale = typical_std * offset_factor
    else:
        offset_scale = (np.nanmean(np.abs(signal_data)) + 1e-3) * offset_factor
        
    if offset_scale == 0:
        offset_scale = offset_factor * 1.0

    # Baseline offsets for each channel (negative for top-down plotting)
    baseline_offsets = np.arange(actual_num_channels) * -offset_scale

    # Apply baseline offsets to the data
    # signal_data shape: (time_points, num_channels)
    # baseline_offsets shape: (num_channels,) -> broadcasting adds offsets to each channel
    offsetted_data = signal_data + baseline_offsets[np.newaxis, :]

    # --- Plotting Signals ---
    ax.plot(time_vector, offsetted_data, lw=0.8)

    # --- MODIFIED: Setting Y-Ticks and Labels (Centered on the *mean* of each plotted trace) ---
    # Calculate the mean y-value of each offsetted channel trace.
    # This effectively gives: mean_of_original_signal + baseline_offset for each channel.
    trace_mean_y_positions = np.nanmean(offsetted_data, axis=0)
    
    # Fallback for any channels that might have been all NaNs (though unlikely for EEG)
    # If a trace_mean is NaN, use its baseline_offset as the tick position.
    final_ytick_positions = np.where(np.isnan(trace_mean_y_positions), 
                                     baseline_offsets, 
                                     trace_mean_y_positions)
    
    ax.set_yticks(final_ytick_positions)
    ax.set_yticklabels(effective_channel_names)
    # --- END OF MODIFICATION ---

    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Channel', fontsize=10)
    ax.set_title(f'{title_prefix}\n(Duration: {actual_time_points/sampling_rate:.2f}s, {actual_num_channels} Ch)', fontsize=12)
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.grid(True, axis='x', linestyle='-', alpha=0.5)
    
    if actual_time_points > 0:
        ax.set_xlim(time_vector[0], time_vector[-1])
    if actual_num_channels > 0:
        y_margin = offset_scale * 0.75 # Visual padding
        # Determine y-limits based on the actual min/max of offsetted data plus margin,
        # or based on the extent of the first and last channel's mean positions plus margin.
        # Using min/max of all offsetted data can be more robust if signals are wild.
        min_y_plot = np.nanmin(offsetted_data) - y_margin
        max_y_plot = np.nanmax(offsetted_data) + y_margin
        ax.set_ylim(min_y_plot, max_y_plot)


    if show_plot_later:
        plt.tight_layout(pad=1.0)
        plt.show()

def plot_signals(signal, title, channels_to_plot=5):
    plt.figure(figsize=(12, 5))
    for i in range(min(channels_to_plot, signal.shape[0])):
        plt.plot(signal[i, :], label=f"Ch {i}")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_training_loss(train_losses, val_losses=None, loss_name="Loss"):
    """Plots the training and optional validation loss over epochs."""
    if not train_losses or not isinstance(train_losses, list):
        print("Invalid training loss data provided.")
        return

    num_epochs_train = len(train_losses)
    if num_epochs_train == 0:
        print("No training loss data to plot.")
        return
        
    epochs_range = range(1, num_epochs_train + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, marker="o", linestyle='-', label="Training " + loss_name)

    if val_losses and isinstance(val_losses, list) and len(val_losses) == num_epochs_train:
        plt.plot(epochs_range, val_losses, marker="o", linestyle='--', label="Validation " + loss_name)
    elif val_losses:
        print("Warning: Validation loss data provided but length mismatched or invalid type. Plotting training loss only.")

    plt.title(f"Training and Validation {loss_name} Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(loss_name, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(ticks=epochs_range[::max(1, num_epochs_train//10)], labels=epochs_range[::max(1, num_epochs_train//10)]) # Show reasonable number of x-ticks
    plt.tight_layout()
    plt.show()