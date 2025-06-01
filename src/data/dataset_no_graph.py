import os
import os.path as osp
from typing import Tuple, List

from scipy import signal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.preprocessing_funcs import time_filtering


class EEGTimeSeriesDataset(Dataset):
    def __init__(
        self,
        root,
        clips: pd.DataFrame,
        signal_folder: str,
        segment_length: int = 3000,
        apply_rereferencing: bool = True,
        apply_normalization: bool = True,
        apply_filtering: bool = True,
        bandpass_frequencies: Tuple[float, float] = (0.5, 50),
        sampling_rate: int = 250,
    ):
        self.root = root
        self.clips = clips
        self.signal_folder = signal_folder
        self.segment_length = segment_length
        self.apply_filtering = apply_filtering
        self.apply_rereferencing = apply_rereferencing
        self.apply_normalization = apply_normalization
        self.sampling_rate = sampling_rate

        self.channels = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"
        ]
        self.n_channels = len(self.channels)

        self.bp_filter = signal.butter(
            4, bandpass_frequencies, btype="bandpass", output="sos", fs=sampling_rate
        )
        notch_filter = signal.iirnotch(w0=60, Q=30, fs=sampling_rate)
        self.notch_filter = signal.tf2sos(*notch_filter)

        self.samples = []
        self._process_sessions()

    def _filter_signal(self, signal):
        return time_filtering(signal, bp_filter=self.bp_filter, notch_filter=self.notch_filter)

    def _rereference(self, signal):
        avg = np.mean(signal, axis=1, keepdims=True)
        return signal - avg

    def _normalize(self, signal):
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        return (signal - mean) / (std + 1e-6)

    def _preprocess_signal(self, signal):
        if self.apply_filtering:
            signal = self._filter_signal(signal)
        if self.apply_rereferencing:
            signal = self._rereference(signal)
        if self.apply_normalization:
            signal = self._normalize(signal)
        return signal

    def _process_sessions(self):
        sessions = list(self.clips.groupby(["patient", "session"]))
        for (_, _), session_df in sessions:
            session_signal = pd.read_parquet(
                f"{self.signal_folder}/{session_df['signals_path'].values[0]}"
            )
            session_signal = self._preprocess_signal(session_signal)

            for _, row in session_df.iterrows():
                start = int(row["start_time"] * self.sampling_rate)
                end = int(row["end_time"] * self.sampling_rate)
                segment_signal = session_signal[start:end].T

                if segment_signal.shape[1] != self.segment_length:
                    continue  # Skip if segment is not of expected length

                x = torch.tensor(segment_signal, dtype=torch.float)  # shape: [C, T]
                # x = x.permute(1, 0)  # shape: [T, C] for CNN-LSTM
                y = torch.tensor([row["label"]], dtype=torch.float)

                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
