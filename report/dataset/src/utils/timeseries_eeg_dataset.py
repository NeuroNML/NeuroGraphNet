import os
import os.path as osp
from pathlib import Path
from typing import Tuple, List, Optional

from scipy import signal
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from seiz_eeg.schemas import ClipsDF

class TimeseriesEEGDataset(Dataset):
    def __init__(
        self,
        clips_df: pd.DataFrame, # DataFrame with clip/segment info
        signal_folder: str,     # Folder for raw signal files (if mode is 'signal')
        # Mode selection (explicitly choose one)
        mode: str = "signal", # "signal", "feature", or "embedding"
        # Feature mode (if mode="feature")
        feature_file_path: Optional[str] = None, # e.g., path to X_train.npy
        # Embedding mode (if mode="embedding")
        embedding_file_path: Optional[str] = None, # e.g., path to embeddings.npy
        labels_for_embedding_file_path: Optional[str] = None, # e.g., path to labels_embeddings.npy
        # Signal processing parameters (if mode="signal")
        segment_length_timesteps: Optional[int] = None, # Required if mode="signal" and clips_df doesn't define fixed duration
        apply_rereferencing: bool = True,
        apply_normalization: bool = True, # Applies to signals, features, and embeddings
        apply_filtering: bool = True,
        bandpass_frequencies: Tuple[float, float] = (0.5, 50),
        notch_freq_hz: Optional[float] = 60.0,
        notch_q_factor: float = 30.0,
        sampling_rate: Optional[int] = None, # Required if mode="signal". Can be read from clips_df too.
        # Caching parameters
        root: Optional[str] = None, # Root directory for cached processed data
        force_reprocess: bool = False, # Force reprocessing and overwrite cached data
        prefetch_data: bool = False, # Load all data into memory
    ):
        print(f"🚀 Initializing TimeseriesEEGDataset in {mode.upper()} mode.")
        self.clips_df = clips_df
        self.signal_folder = Path(signal_folder)
        self.mode = mode.lower()
        
        # Caching setup
        self.root = Path(root) if root else None
        self.force_reprocess = force_reprocess
        self.prefetch_data = prefetch_data
        self._data_list = None  # For prefetching

        self.apply_rereferencing = apply_rereferencing
        self.apply_normalization = apply_normalization
        self.apply_filtering = apply_filtering
        self.bandpass_frequencies = bandpass_frequencies
        self.notch_freq_hz = notch_freq_hz
        self.notch_q_factor = notch_q_factor
        
        self.channels = [
            "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"
        ]
        self.n_channels = len(self.channels)
        
        # Initialize samples list and processed items count
        self.samples = []
        self._processed_items_count = 0

        if self.mode == "signal":
            if sampling_rate is None:
                if ClipsDF.sampling_rate not in clips_df.columns:
                    raise ValueError(f"sampling_rate must be provided or column '{ClipsDF.sampling_rate}' must be in clips_df for signal mode.")
                unique_sr = clips_df[ClipsDF.sampling_rate].unique()
                if len(unique_sr) > 1:
                    raise ValueError("Multiple sampling rates found in clips_df. Expected a single sampling rate for signal mode.")
                self.sampling_rate = unique_sr[0]
            else:
                self.sampling_rate = sampling_rate
            print(f"   - Sampling rate: {self.sampling_rate} Hz")

            if segment_length_timesteps is None:
                 # Try to derive from clips_df if start/end times are consistent
                if ClipsDF.start_time in clips_df.columns and ClipsDF.end_time in clips_df.columns:
                    durations_ts = np.unique(np.round((clips_df[ClipsDF.end_time] - clips_df[ClipsDF.start_time]) * self.sampling_rate))
                    if len(durations_ts) == 1:
                        self.segment_length_timesteps = int(durations_ts[0])
                        print(f"   - Derived segment length: {self.segment_length_timesteps} timesteps.")
                    else:
                        raise ValueError("segment_length_timesteps must be provided if clips have variable durations.")
                else:
                    raise ValueError("segment_length_timesteps must be provided, or clips_df needs start_time, end_time, and sampling_rate to derive it.")
            else:
                 self.segment_length_timesteps = segment_length_timesteps
            print(f"   - Segment length: {self.segment_length_timesteps} timesteps.")
            
            # Check if labels are available (they might not be for test sets)
            self.has_labels = ClipsDF.label in self.clips_df.columns
            if not self.has_labels:
                print(f"   ⚠️ Info: Column '{ClipsDF.label}' not found in clips_df. Processing without labels (e.g., test set).")
            
            if self.apply_filtering:
                self.bp_filter_coeffs = signal.butter(4, self.bandpass_frequencies, btype="bandpass",
                                                      output="sos", fs=self.sampling_rate)
                if self.notch_freq_hz is not None:
                    notch_coeffs_ba = signal.iirnotch(w0=self.notch_freq_hz, Q=self.notch_q_factor, fs=self.sampling_rate)
                    self.notch_filter_coeffs = signal.tf2sos(*notch_coeffs_ba)
                else:
                    self.notch_filter_coeffs = None
            
            # Set up caching prefix and check for existing cache
            self._current_file_prefix = self._determine_prefix_for_mode()
            
            if self.root and self._should_use_cache():
                print(f"   ✅ Using existing cached data from {self.processed_dir}")
                self._processed_items_count = self._count_existing_processed_files()
            else:
                if self.root and self.force_reprocess:
                    print(f"   🔄 Force reprocessing enabled, clearing existing cache...")
                    self._clear_processed_files()
                self._process_sessions()

        elif self.mode == "feature":
            if feature_file_path is None:
                raise ValueError("feature_file_path must be provided for feature mode.")
            self.feature_data = np.load(feature_file_path)
            # Labels for features are expected to be in clips_df, aligned by index.
            if len(self.feature_data) != len(self.clips_df):
                 print(f"   ⚠️ Warning: Feature data length ({len(self.feature_data)}) "
                       f"differs from clips_df length ({len(self.clips_df)}). Label alignment might be incorrect.")
            
            # Check if labels are available (they might not be for test sets)
            self.has_labels = ClipsDF.label in self.clips_df.columns
            if not self.has_labels:
                print(f"   ⚠️ Info: Column '{ClipsDF.label}' not found in clips_df. Processing without labels (e.g., test set).")
            
            # Set up caching prefix and check for existing cache
            self._current_file_prefix = self._determine_prefix_for_mode()
            
            if self.root and self._should_use_cache():
                print(f"   ✅ Using existing cached data from {self.processed_dir}")
                self._processed_items_count = self._count_existing_processed_files()
            else:
                if self.root and self.force_reprocess:
                    print(f"   🔄 Force reprocessing enabled, clearing existing cache...")
                    self._clear_processed_files()
                self._process_features()

        elif self.mode == "embedding":
            if embedding_file_path is None:
                raise ValueError("embedding_file_path must be provided for embedding mode.")
            self.embedding_data = np.load(embedding_file_path)
            
            # Labels are optional for test sets
            if labels_for_embedding_file_path is not None:
                self.embedding_labels = np.load(labels_for_embedding_file_path)
                if len(self.embedding_data) != len(self.embedding_labels):
                    raise ValueError("Mismatch between length of embedding data and embedding labels.")
                self.has_labels = True
            else:
                self.embedding_labels = None
                self.has_labels = False
                print(f"   ⚠️ Info: No labels provided for embeddings. Processing without labels (e.g., test set).")
                
            if self.embedding_data.ndim !=3 or self.embedding_data.shape[1] != self.n_channels: # Expect [N, C, D_embed]
                 raise ValueError(f"Embedding data shape unexpected: {self.embedding_data.shape}. Expected [N, {self.n_channels}, Dimensions].")
            
            # Set up caching prefix and check for existing cache
            self._current_file_prefix = self._determine_prefix_for_mode()
            
            if self.root and self._should_use_cache():
                print(f"   ✅ Using existing cached data from {self.processed_dir}")
                self._processed_items_count = self._count_existing_processed_files()
            else:
                if self.root and self.force_reprocess:
                    print(f"   🔄 Force reprocessing enabled, clearing existing cache...")
                    self._clear_processed_files()
                self._process_embeddings()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'signal', 'feature', or 'embedding'.")
        
        # Handle prefetching if requested
        if self.prefetch_data and self.root and self._processed_items_count > 0:
            self._prefetch_data()
        
        final_count = self._processed_items_count if self.root and self._processed_items_count > 0 else len(self.samples)
        print(f"🏁 TimeseriesEEGDataset initialization complete. Loaded {final_count} samples.")

    def _filter_signal(self, eeg_data_time_channels: np.ndarray) -> np.ndarray:
        # Input: [Time, Channels]
        filtered_data = eeg_data_time_channels
        if self.apply_filtering and hasattr(self, 'bp_filter_coeffs'): # Check if filters are initialized
            filtered_data = signal.sosfiltfilt(self.bp_filter_coeffs, filtered_data, axis=0)
            if hasattr(self, 'notch_filter_coeffs') and self.notch_filter_coeffs is not None:
                filtered_data = signal.sosfiltfilt(self.notch_filter_coeffs, filtered_data, axis=0)
        return filtered_data

    def _rereference_signal(self, eeg_data_time_channels: np.ndarray) -> np.ndarray:
        # Input: [Time, Channels]
        # Common average rereferencing
        if eeg_data_time_channels.shape[1] > 1: # More than one channel
            avg = np.mean(eeg_data_time_channels, axis=1, keepdims=True)
            return eeg_data_time_channels - avg
        return eeg_data_time_channels # No rereferencing if single channel

    def _normalize_signal(self, eeg_data_time_channels: np.ndarray) -> np.ndarray:
        # Input: [Time, Channels]
        # Normalize each channel's time series independently (Z-score)
        mean = np.mean(eeg_data_time_channels, axis=0, keepdims=True)
        std = np.std(eeg_data_time_channels, axis=0, keepdims=True) + 1e-6
        return (eeg_data_time_channels - mean) / std

    def _preprocess_raw_signal(self, eeg_data_time_channels: np.ndarray) -> np.ndarray:
        # Input: [Time, Channels]
        processed_data = eeg_data_time_channels.copy()
        if self.apply_filtering:
            processed_data = self._filter_signal(processed_data)
        if self.apply_rereferencing:
            processed_data = self._rereference_signal(processed_data)
        if self.apply_normalization: # Normalization for raw signals
            processed_data = self._normalize_signal(processed_data)
        return processed_data

    def _process_embeddings(self):
        # self.embedding_data is [N, C, D_embed]
        # self.embedding_labels is [N]
        print(f"   - Processing {len(self.embedding_data)} embeddings...")
        log_interval = max(1, len(self.embedding_data) // 10)
        
        for i in range(len(self.embedding_data)):
            if (i + 1) % log_interval == 0 or i == len(self.embedding_data) - 1:
                print(f"     - Processing embedding {i + 1}/{len(self.embedding_data)}...")
            
            # segment_embedding is [C, D_embed]
            segment_embedding = self.embedding_data[i].copy()
            
            if self.apply_normalization:
                # Normalize each channel's embedding vector independently
                mean = segment_embedding.mean(axis=1, keepdims=True) # Mean over D_embed for each channel
                std = segment_embedding.std(axis=1, keepdims=True) + 1e-6
                segment_embedding = (segment_embedding - mean) / std

            # Handle labels: create label tensor only if labels are available
            if self.has_labels:
                y = torch.tensor([self.embedding_labels[i]], dtype=torch.float)
            else:
                y = None  # No labels for test set
                
            # Output x: can be [C, D_embed] or flattened.
            # Original dataset_no_graph.py flattened it. Let's keep that for typical non-GNN models.
            x = torch.tensor(segment_embedding.flatten(), dtype=torch.float32) # Flattened: [C * D_embed]
            # If a model expects [C, D_embed], use:
            # x = torch.tensor(segment_embedding, dtype=torch.float32)
            
            if self.root:
                self._save_processed_item(x, y, i)
                self._processed_items_count += 1
            else:
                self.samples.append((x, y))
        
        print(f"   ✅ Processed {len(self.embedding_data)} embeddings.")

    def _process_features(self):
        # self.feature_data is [N, C*F_feat_per_channel] or [N, C, F_feat_per_channel]
        print(f"   - Processing {len(self.feature_data)} feature sets...")
        log_interval = max(1, len(self.feature_data) // 10)
        processed_count = 0
        
        for index, features_for_sample_raw in enumerate(self.feature_data):
            if (index + 1) % log_interval == 0 or index == len(self.feature_data) - 1:
                print(f"     - Processing feature set {index + 1}/{len(self.feature_data)}...")
            
            try:
                # Reshape to [Channels, Features_per_channel]
                if features_for_sample_raw.ndim == 1: # [C*F]
                    segment_features_shaped = features_for_sample_raw.reshape(self.n_channels, -1)
                elif features_for_sample_raw.ndim == 2 and features_for_sample_raw.shape[0] == self.n_channels: # Already [C,F]
                    segment_features_shaped = features_for_sample_raw
                else:
                    print(f"   ⚠️ Feature set {index} has unexpected shape {features_for_sample_raw.shape}. Expected [C*F] or [C,F]. Skipping.")
                    continue
            except ValueError as e:
                print(f"   ⚠️ Error reshaping features for sample {index}: {e}. Original shape {features_for_sample_raw.shape}. Skipping.")
                continue

            segment_features = segment_features_shaped.copy() # Now [C, F_feat_per_channel]

            if self.apply_normalization:
                # Normalize each channel's feature vector independently
                mean = segment_features.mean(axis=1, keepdims=True) # Mean over F_feat for each channel
                std = segment_features.std(axis=1, keepdims=True) + 1e-6
                segment_features = (segment_features - mean) / std
            
            if index >= len(self.clips_df):
                print(f"   ⚠️ Feature index {index} out of bounds for clips_df (len {len(self.clips_df)}), cannot get metadata. Skipping.")
                continue
            
            # Handle labels: create label tensor only if labels are available
            if self.has_labels:
                label_val = self.clips_df[ClipsDF.label].iloc[index]
                y = torch.tensor([label_val], dtype=torch.float)
            else:
                y = None  # No labels for test set

            # Output x: Flattened [C * F_feat_per_channel]
            x = torch.tensor(segment_features.flatten(), dtype=torch.float32)
            
            if self.root:
                self._save_processed_item(x, y, processed_count)
                self._processed_items_count += 1
            else:
                self.samples.append((x, y))
            
            processed_count += 1
        
        print(f"   ✅ Processed {processed_count} feature sets.")

    def _process_sessions(self):
        # Iterate through clips_df, load signal for each segment, process, and store
        # This is simpler than group-by-session if signals_path is specific to each segment's file
        # or if files are small enough that reloading isn't a huge overhead.
        # For large session files with many segments, dataset_no_graph's original group-by was more efficient.
        # Let's offer a simplified row-by-row processing first.
        
        print(f"   - Processing {len(self.clips_df)} segments from clips_df row by row...")
        log_interval = max(1, len(self.clips_df) // 10)
        processed_count = 0

        for row_num, (idx, row_data) in enumerate(self.clips_df.iterrows()):
            if (row_num + 1) % log_interval == 0 or row_num == len(self.clips_df) - 1:
                print(f"     - Processing clip/segment {row_num + 1}/{len(self.clips_df)}...")

            signal_file_name = row_data[ClipsDF.signals_path]
            full_signal_path = self.signal_folder / signal_file_name
            if not full_signal_path.exists(): # Try path as is
                full_signal_path = Path(signal_file_name)
                if not full_signal_path.exists():
                    print(f"     ⚠️ Signal file {signal_file_name} not found. Skipping segment {idx}.")
                    continue
            try:
                session_signal_df = pd.read_parquet(full_signal_path)
                if not all(ch in session_signal_df.columns for ch in self.channels):
                    print(f"     ⚠️ Not all expected channels found in {full_signal_path}. Skipping segment {idx}.")
                    continue
                
                # Ensure correct channel order and get values [Time, Channels]
                session_signal_values = session_signal_df[self.channels].values 
                
                start_idx = int(row_data[ClipsDF.start_time] * self.sampling_rate)
                end_idx = int(row_data[ClipsDF.end_time] * self.sampling_rate)
                
                # Ensure segment length consistency using self.segment_length_timesteps
                # If end_idx - start_idx is not self.segment_length_timesteps, adjust or skip
                # This assumes clips_df start/end times define segments of self.segment_length_timesteps

                if (end_idx - start_idx) != self.segment_length_timesteps:
                    # This could be due to rounding or variable clip lengths not meant for this fixed-length setup
                    # Option 1: adjust end_idx to match fixed length if possible
                    # Option 2: skip if not matching (safer for now)
                    # print(f"     ⚠️ Segment {idx} from {full_signal_path} has duration {end_idx-start_idx} timesteps, expected {self.segment_length_timesteps}. Adjusting end_idx.")
                    # end_idx = start_idx + self.segment_length_timesteps # This might go out of bounds
                    pass # Allow variable length segments if self.segment_length_timesteps wasn't strictly enforced or used for padding later

                if end_idx > session_signal_values.shape[0]:
                    print(f"     ⚠️ Segment {idx} end_idx {end_idx} exceeds signal length {session_signal_values.shape[0]}. Clamping.")
                    end_idx = session_signal_values.shape[0]
                if start_idx >= end_idx:
                    print(f"     ⚠️ Segment {idx} start_idx {start_idx} >= end_idx {end_idx}. Skipping.")
                    continue

                segment_signal_time_channels = session_signal_values[start_idx:end_idx, :]

                # Check actual length against expected fixed length
                if segment_signal_time_channels.shape[0] != self.segment_length_timesteps:
                    # Handle segments not matching the expected length (e.g. pad, truncate, or skip)
                    # For now, skip if not exact match, common for some models.
                    print(f"     ⚠️ Segment {idx} actual length {segment_signal_time_channels.shape[0]} != expected {self.segment_length_timesteps}. Skipping.")
                    continue
                
                processed_segment = self._preprocess_raw_signal(segment_signal_time_channels) # Output: [Time, Channels]
                
                # For typical time series models (CNN, LSTM), input is often [Channels, Time]
                x_tensor_data = processed_segment.T # Transpose to [Channels, Time]
                
                x = torch.tensor(x_tensor_data, dtype=torch.float32)
                
                # Handle labels: create label tensor only if labels are available
                if self.has_labels:
                    y = torch.tensor([row_data[ClipsDF.label]], dtype=torch.float)
                else:
                    y = None  # No labels for test set
                
                if self.root:
                    self._save_processed_item(x, y, processed_count)
                    self._processed_items_count += 1
                else:
                    self.samples.append((x, y))
                
                processed_count += 1

            except Exception as e:
                print(f"     ⚠️ Error processing segment {idx} from {full_signal_path}: {e}. Skipping.")
                continue
        
        print(f"   ✅ Processed {processed_count} segments from raw signals.")
    
    def __len__(self):
        if self.root and self._processed_items_count > 0:
            return self._processed_items_count
        return len(self.samples)

    def __getitem__(self, idx):
        if self.root and self._processed_items_count > 0:
            return self._get_cached_item(idx)
        return self.samples[idx]
    
    def _determine_prefix_for_mode(self) -> str:
        """Determine the file prefix based on the current mode."""
        if self.mode == "embedding":
            return "ts_embed_"
        elif self.mode == "feature":
            return "ts_feat_"
        return "ts_signal_"  # Default for signal mode
    
    @property
    def processed_dir(self) -> str:
        """Directory where processed data will be cached."""
        if not self.root:
            raise ValueError("Root directory not set for caching")
        return osp.join(str(self.root), "processed")
    
    def _get_cached_item(self, idx: int):
        """Load a cached item from disk."""
        if not (0 <= idx < self._processed_items_count):
            raise IndexError(f"Index {idx} out of range [0, {self._processed_items_count})")
        
        if hasattr(self, '_data_list') and self._data_list is not None:
            return self._data_list[idx]
        
        file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{idx}.pt')
        try:
            data = torch.load(file_path)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Cached file not found: {file_path}")
        except Exception as e:
            # Handle PyTorch 2.6+ weights_only error
            if "weights_only" in str(e).lower():
                data = torch.load(file_path, weights_only=False)
                return data
            else:
                raise RuntimeError(f"Error loading cached file {file_path}: {e}")
    
    def _save_processed_item(self, x: torch.Tensor, y: Optional[torch.Tensor], idx: int):
        """Save a processed item to disk."""
        if not self.root:
            return  # No caching if root not set
        
        os.makedirs(self.processed_dir, exist_ok=True)
        file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{idx}.pt')
        torch.save((x, y), file_path)
    
    def _count_existing_processed_files(self) -> int:
        """Count existing processed files with the current prefix."""
        if not self.root or not osp.exists(self.processed_dir):
            return 0
        
        count = 0
        while True:
            file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{count}.pt')
            if osp.exists(file_path):
                count += 1
            else:
                break
        return count
    
    def _clear_processed_files(self):
        """Clear existing processed files with the current prefix."""
        if not self.root or not osp.exists(self.processed_dir):
            return
        
        count = 0
        while True:
            file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{count}.pt')
            if osp.exists(file_path):
                os.remove(file_path)
                count += 1
            else:
                break
        print(f"   - Cleared {count} existing cached files with prefix '{self._current_file_prefix}'")
    
    def _should_use_cache(self) -> bool:
        """Check if we should use cached data instead of processing."""
        if not self.root or self.force_reprocess:
            return False
        
        existing_count = self._count_existing_processed_files()
        expected_count = len(self.clips_df) if self.mode == "signal" else (
            len(self.feature_data) if hasattr(self, 'feature_data') else
            len(self.embedding_data) if hasattr(self, 'embedding_data') else 0
        )
        
        return existing_count == expected_count and existing_count > 0
    
    def _prefetch_data(self):
        """Load all cached data into memory."""
        if not self.root or self._processed_items_count == 0:
            print("   ⚠️ No cached data to prefetch")
            return
        
        self._data_list = [None] * self._processed_items_count
        print(f"🚀 Starting prefetch of {self._processed_items_count} items for {self.mode} mode...")
        
        prefetched_count = 0
        log_interval = max(1, self._processed_items_count // 10)
        
        for i in range(self._processed_items_count):
            try:
                file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{i}.pt')
                try:
                    self._data_list[i] = torch.load(file_path)
                except Exception as e:
                    # Handle PyTorch 2.6+ weights_only error
                    if "weights_only" in str(e).lower():
                        self._data_list[i] = torch.load(file_path, weights_only=False)
                    else:
                        raise
                prefetched_count += 1
                
                if (i + 1) % log_interval == 0 or i == self._processed_items_count - 1:
                    print(f"   - Prefetched {i + 1}/{self._processed_items_count} items")
            except Exception as e:
                print(f"   ⚠️ Error prefetching item {i}: {e}")
                self._data_list[i] = None
        
        # Filter out failed loads
        loaded_items = [item for item in self._data_list if item is not None]
        if len(loaded_items) != prefetched_count:
            print(f"   ⚠️ Some files failed to load. Expected {prefetched_count}, got {len(loaded_items)}")
        
        self._data_list = loaded_items
        print(f"🏁 Prefetching complete. Loaded {len(self._data_list)} items into memory.")