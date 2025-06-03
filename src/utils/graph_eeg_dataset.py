import os.path as osp
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Callable

import pandas as pd
import numpy as np
from scipy import signal
from seiz_eeg.schemas import ClipsDF

import torch
from torch_geometric.data import Dataset, Data
from src.utils.signal import rereference, normalize

class GraphEEGDataset(Dataset):
    def __init__(
        self,
        root: str,
        clips_df: pd.DataFrame,
        signal_folder: str,
        # Raw signal mode is default if other sources are not specified/used
        # Feature mode
        extracted_features_array: Optional[np.ndarray] = None, # Pass the numpy array directly
        use_extracted_features: bool = False,
        # Embedding mode
        embedding_file_path: Optional[str] = None, # Path to .npy file for embeddings
        labels_for_embedding_file_path: Optional[str] = None, # Path to .npy file for labels of embeddings
        use_embeddings: bool = False,
        # Edge strategy
        edge_strategy: str = "spatial",
        spatial_distance_file: Optional[str] = None,
        correlation_threshold: float = 0.7,
        top_k_correlation: Optional[int] = None,
        # Processing params
        force_reprocess: bool = False,
        bandpass_frequencies: Tuple[float, float] = (0.5, 50.0),
        notch_freq_hz: Optional[float] = 60.0,
        notch_q_factor: float = 30.0,
        apply_filtering: bool = True, # For raw signals
        apply_rereferencing: bool = True, # For raw signals
        apply_normalization: bool = True, # For raw signals, features, and embeddings
        # Other
        prefetch_data: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.clips_df = clips_df
        self.signal_folder = Path(signal_folder)
        self.force_reprocess_flag = force_reprocess

        # Determine processing mode
        self.mode = "signal" # Default
        if use_embeddings and embedding_file_path and labels_for_embedding_file_path:
            self.mode = "embedding"
            print("🚀 Initializing GraphEEGDataset in EMBEDDING mode.")
            self.embedding_data = np.load(embedding_file_path)
            self.embedding_labels = np.load(labels_for_embedding_file_path)
            if len(self.embedding_data) != len(self.embedding_labels):
                raise ValueError("Mismatch between length of embedding data and embedding labels.")
            if self.embedding_data.ndim != 3 or self.embedding_data.shape[1] == 0: # Expect [N, C, D_embed]
                 raise ValueError(f"Embedding data shape unexpected: {self.embedding_data.shape}. Expected [N, Channels, Dimensions].")
            self.n_channels_data = self.embedding_data.shape[1]
        elif use_extracted_features and extracted_features_array is not None:
            self.mode = "feature"
            print("🚀 Initializing GraphEEGDataset in FEATURE mode.")
            self.node_features_source = extracted_features_array
            if self.node_features_source.ndim != 2 and self.node_features_source.ndim != 3 : # Expect [N_segments, N_channels * N_features_per_channel] or [N_segments, N_channels, N_features_per_channel]
                 raise ValueError(f"Extracted features array shape unexpected: {self.node_features_source.shape}.")
            # We will reshape to [N_segments, N_channels, N_features_per_channel] in processing if needed
        else:
            print("🚀 Initializing GraphEEGDataset in SIGNAL mode.")
            self.mode = "signal"

        print(f"   - Root: {self.root}")
        print(f"   - Mode: {self.mode.upper()}")

        self.edge_strategy = edge_strategy
        self.spatial_distance_file = spatial_distance_file
        self.correlation_threshold = correlation_threshold
        self.top_k_correlation = top_k_correlation
        print(f"   - Edge strategy: {self.edge_strategy}")

        self.apply_filtering = apply_filtering
        self.apply_rereferencing = apply_rereferencing
        self.apply_normalization = apply_normalization
        self.bandpass_frequencies = bandpass_frequencies
        self.notch_freq_hz = notch_freq_hz
        self.notch_q_factor = notch_q_factor
        if self.mode == "signal":
            print(f"   - Raw Signal Preprocessing: Filtering={apply_filtering}, Rereferencing={apply_rereferencing}, Normalization={apply_normalization}")
        elif self.mode == "feature" or self.mode == "embedding":
            print(f"   - Node Feature Normalization: {apply_normalization}")


        if not isinstance(clips_df, pd.DataFrame) and self.mode == "signal":
            raise TypeError("'clips_df' must be a pandas DataFrame for signal mode.")

        if self.mode == "signal":
            if ClipsDF.sampling_rate not in clips_df.columns:
                raise ValueError(f"Column '{ClipsDF.sampling_rate}' not found in clips_df for signal mode.")
            unique_sampling_rates = clips_df[ClipsDF.sampling_rate].unique()
            if len(unique_sampling_rates) > 1:
                raise ValueError("Multiple sampling rates found in clips_df. Dataset expects a single sampling rate for signal mode.")
            self.sampling_rate = unique_sampling_rates[0]
            print(f"   - Detected sampling rate: {self.sampling_rate} Hz")

            if ClipsDF.end_time in clips_df.columns and ClipsDF.start_time in clips_df.columns:
                _lengths = np.unique(self.clips_df[ClipsDF.end_time] - self.clips_df[ClipsDF.start_time])
                self.clip_duration = _lengths[0] if np.allclose(_lengths, _lengths[0]) else -1
                if self.clip_duration <= 0:
                    print(f"   ⚠️ Warning: Could not determine a consistent clip_duration or it's non-positive ({self.clip_duration}).")
                self.segment_length_timesteps = round(self.clip_duration * self.sampling_rate)
                print(f"   - Clip duration: {self.clip_duration}s, Segment length: {self.segment_length_timesteps} timesteps")
            else:
                print(f"   ⚠️ Warning: Columns '{ClipsDF.start_time}' or '{ClipsDF.end_time}' not found. Cannot determine clip_duration or segment_length_timesteps.")
                self.clip_duration = -1
                self.segment_length_timesteps = -1
        else: # Feature or Embedding mode
            self.sampling_rate = None # Not directly used from clips_df
            self.clip_duration = None
            self.segment_length_timesteps = None


        # Channel definition is important for edge creation and data reshaping
        self.channels = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
                           "F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"]
        self.n_channels = len(self.channels)
        
        if self.mode == "embedding" and self.embedding_data.shape[1] != self.n_channels:
            raise ValueError(f"Embedding data has {self.embedding_data.shape[1]} channels, but class expects {self.n_channels}.")
        # For feature mode, n_channels will be checked during processing when reshaping.


        if self.mode == "signal" and self.apply_filtering:
            print(f"   - Designing filters: Bandpass {self.bandpass_frequencies} Hz, Notch {self.notch_freq_hz} Hz")
            self.bp_filter_coeffs = signal.butter(4, self.bandpass_frequencies, btype="bandpass",
                                                  output="sos", fs=self.sampling_rate)
            if self.notch_freq_hz is not None and self.sampling_rate is not None:
                notch_coeffs_ba = signal.iirnotch(w0=self.notch_freq_hz, Q=self.notch_q_factor, fs=self.sampling_rate)
                self.notch_filter_coeffs = signal.tf2sos(*notch_coeffs_ba)
            else:
                self.notch_filter_coeffs = None

        self.spatial_distances_map = None
        if self.edge_strategy == "spatial":
            if self.spatial_distance_file is None:
                raise ValueError("spatial_distance_file must be provided for 'spatial' edge strategy.")
            self.spatial_distances_map = self._load_spatial_distances()

        self._processed_items_count = 0
        self._current_file_prefix = self._determine_prefix_for_mode()
        
        # Call super().__init__ before accessing processed_dir or calling process()
        # The transform, pre_transform, pre_filter are PyG's arguments
        super().__init__(str(self.root), transform, pre_transform, pre_filter) # root needs to be string for PyG
        
        # Determine initial _processed_items_count if not force_reprocessing
        # This count is based on files matching the CURRENT mode's prefix
        if not self.force_reprocess_flag and osp.exists(self.processed_dir):
            try:
                processed_files = [f for f in os.listdir(self.processed_dir)
                                   if f.startswith(self._current_file_prefix) and f.endswith(".pt")]
                self._processed_items_count = len(processed_files)
                if self._processed_items_count > 0:
                    print(f"   - Found {self._processed_items_count} existing processed files for {self.mode} mode.")
            except FileNotFoundError: # processed_dir might not exist yet if raw_dir also doesn't
                 if not osp.exists(self.raw_dir) and not osp.exists(self.processed_dir):
                     # This condition implies that it's fine if processed_dir doesn't exist yet,
                     # especially if process() will create it.
                     pass
                 else:
                     # If processed_dir was expected (e.g. raw_dir exists), then re-raise or warn.
                     print(f"   ⚠️ Warning: Processed directory {self.processed_dir} not found during init scan.")


        self._data_list = None # For prefetching
        if prefetch_data:
            # Prefetching should ideally happen after process() or if files already exist
            # If process() is called by PyG automatically, prefetch here might be too early
            # or rely on already processed files.
            # For now, let's assume prefetch is called after data is confirmed to be processed.
            if self._processed_items_count > 0 : # Only prefetch if files are known to exist
                 self._prefetch_data()
            else:
                 print("   - Prefetching skipped: No processed items count determined yet or count is zero.")
        
        print(f"🏁 GraphEEGDataset initialization complete. Current mode: {self.mode.upper()}. Known processed items: {self.len()}")


    def _determine_prefix_for_mode(self) -> str:
        if self.mode == "embedding": return "data_embed_"
        elif self.mode == "feature": return "data_feat_"
        return "data_segment_" # Default for signal mode

    @property
    def raw_dir(self) -> str:
        return osp.join(str(self.root), 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(str(self.root), 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # This tells PyG what to look for in raw_dir to decide if download/processing is needed.
        # It's more of a conceptual list. If these are not found, process() is called.
        if self.mode == "embedding":
            # Check for the source .npy files if they were defining for this mode
            return [Path(self.embedding_file_path).name, Path(self.labels_for_embedding_file_path).name] if self.embedding_file_path and self.labels_for_embedding_file_path else ["embedding_source.npy"]
        elif self.mode == "feature":
            # For features passed as array, raw files aren't "downloaded" in the same way.
            # We can return a placeholder or a name if features were from a conceptual raw file.
            return ["extracted_features_placeholder.npy"] # Indicates features are provided not from raw_dir files
        else: # Signal mode
            if ClipsDF.signals_path in self.clips_df.columns:
                # Unique signal files from the clips_df that would reside in raw_dir
                # This assumes signal_folder is inside self.raw_dir or signals_path are relative to raw_dir
                # For simplicity, if signal_folder is absolute, this property might not be fully representative.
                # Let's assume signals_path are just filenames.
                unique_signal_files = self.clips_df[ClipsDF.signals_path].unique().tolist()
                return [str(Path(f).name) for f in unique_signal_files if f]
        return []


    @property
    def processed_file_names(self) -> List[str]:
        # This should list the files `get(idx)` will try to load.
        # The length of this list is often used by PyG as dataset length.
        # This should be based on successfully processed items.
        return [f'{self._current_file_prefix}{i}.pt' for i in range(self._processed_items_count)]

    @property
    def labels(self) -> Optional[pd.Series]:
        if self.mode == "embedding":
            return pd.Series(self.embedding_labels) if self.embedding_labels is not None else None
        elif self.mode == "feature":
            # Assuming labels for features come from clips_df if it's provided and aligned
            # This part needs careful alignment logic if clips_df is used for labels with features.
            # For now, let's assume labels are part of the Data object's 'y'
            # and this property is more for general dataset characteristic.
            if ClipsDF.label in self.clips_df.columns and len(self.clips_df) == len(self.node_features_source):
                 return self.clips_df[ClipsDF.label]
            return None # Or handle labels differently for features if not from clips_df
        else: # Signal mode
            if ClipsDF.label in self.clips_df.columns:
                return self.clips_df[ClipsDF.label]
        return None

    def process(self):
        print(f"⚙️ process() called for {self.mode.upper()} mode. Target processed directory: {self.processed_dir}")
        
        if not osp.exists(self.raw_dir) and (self.mode == "signal" or self.mode == "embedding"): # Features are passed in memory
            # For signal/embedding mode, if raw_dir is expected to contain something (e.g. original .npy for embeddings, or .parquet for signals)
            # and it doesn't exist, we might need to halt or skip, unless raw_file_names handles this.
            # PyG calls process() if raw_files (from raw_file_names) are not found in raw_dir OR processed_files are not in processed_dir.
            # So, if raw_file_names is accurate, this check might be redundant.
            # If embedding files or signal files are not in raw_dir, PyG would call process().
            # Our raw_file_names for embedding mode checks for embedding_file_path.name, not in raw_dir.
            # This part of PyG logic can be tricky. Let's assume process() is called correctly.
             pass


        if self.force_reprocess_flag:
            print(f"   - Force reprocess: Deleting existing files in {self.processed_dir} with prefix '{self._current_file_prefix}'...")
            deleted_count = 0
            if osp.exists(self.processed_dir):
                for f_name in os.listdir(self.processed_dir):
                    if f_name.startswith(self._current_file_prefix) and f_name.endswith('.pt'):
                        try:
                            os.remove(osp.join(self.processed_dir, f_name))
                            deleted_count +=1
                        except OSError as e:
                            print(f"     ⚠️ Error deleting file {f_name}: {e}")

            print(f"     - Deleted {deleted_count} existing .pt files with prefix '{self._current_file_prefix}'.")
            self._processed_items_count = 0 # Reset count as files are deleted

        if not osp.exists(self.processed_dir):
            print(f"   - Creating processed directory: {self.processed_dir}")
            os.makedirs(self.processed_dir, exist_ok=True)

        # Update prefix just in case mode changed, though mode is set in init
        self._current_file_prefix = self._determine_prefix_for_mode()

        if self.mode == "embedding":
            print("   - Starting processing from pre-computed embeddings...")
            self._process_from_embeddings()
        elif self.mode == "feature":
            print("   - Starting processing from pre-extracted features...")
            self._process_from_features()
        else: # signal
            print("   - Starting processing from raw signal sessions...")
            self._process_from_sessions()
        
        # After processing, self._processed_items_count should be updated by the respective methods.
        print(f"🏁 process() finished. Total items processed and saved in this run for {self.mode} mode: {self._processed_items_count}")


    def _process_from_embeddings(self):
        save_idx = 0
        num_total_embeddings = len(self.embedding_data)
        print(f"   - Total embeddings to process: {num_total_embeddings}")
        log_interval = max(1, num_total_embeddings // 10)

        for i in range(num_total_embeddings):
            if (i + 1) % log_interval == 0 or i == num_total_embeddings -1 :
                print(f"     - Processing embedding item {i + 1}/{num_total_embeddings}...")
            
            # embedding_sample should be [Channels, Dimensions_per_channel]
            embedding_sample = self.embedding_data[i] # Expected shape [C, D_embed]
            if embedding_sample.shape[0] != self.n_channels:
                print(f"     ⚠️ Embedding sample {i} has {embedding_sample.shape[0]} channels, expected {self.n_channels}. Skipping.")
                continue

            node_x_features = embedding_sample.copy()

            if self.apply_normalization:
                # Normalize each channel's embedding vector independently
                mean = node_x_features.mean(axis=1, keepdims=True)
                std = node_x_features.std(axis=1, keepdims=True) + 1e-6
                node_x_features = (node_x_features - mean) / std
            
            x_tensor = torch.tensor(node_x_features, dtype=torch.float)
            edge_index = self._create_edges(node_x_features) # Pass [C, D_embed]

            if edge_index.numel() == 0 and edge_index.shape[0] != 2: # Check for empty or malformed edge_index
                print(f"     ⚠️ No edges created for embedding item {i}. Skipping.")
                continue

            label_val = self.embedding_labels[i]
            y_tensor = torch.tensor([label_val], dtype=torch.float) # Assuming labels are single float values

            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, original_idx=torch.tensor([i]))
            save_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{save_idx}.pt')
            torch.save(data, save_path)
            save_idx += 1
        
        self._processed_items_count = save_idx
        print(f"   ✅ Processed and saved {self._processed_items_count} items from embeddings.")


    def _process_from_features(self):
        save_idx = 0
        # node_features_source could be [N, C*F] or [N, C, F]
        # We need to reshape to [N, C, F] if it's flat, then iterate N
        
        num_total_features_sets = len(self.node_features_source)
        print(f"   - Total feature sets to process: {num_total_features_sets}")
        log_interval = max(1, num_total_features_sets // 10)

        # Determine labels source
        has_labels_in_clips_df = ClipsDF.label in self.clips_df.columns and \
                                 len(self.clips_df) == num_total_features_sets
        if not has_labels_in_clips_df:
             print(f"   ⚠️ Warning: Labels for features will be None or use a default. "
                   f"'{ClipsDF.label}' column not in clips_df or length mismatch "
                   f"({len(self.clips_df)} vs {num_total_features_sets}).")


        for feature_set_idx, features_per_segment_raw in enumerate(self.node_features_source):
            if (feature_set_idx + 1) % log_interval == 0 or feature_set_idx == num_total_features_sets -1 :
                print(f"     - Processing feature item {feature_set_idx + 1}/{num_total_features_sets}...")
            
            try:
                # Reshape to [Channels, Features_per_channel]
                # If features_per_segment_raw is already [C,F], reshape is benign if -1 is used correctly.
                # If it's [C*F], it reshapes.
                if features_per_segment_raw.ndim == 1: # [C*F]
                    node_x_features_shaped = features_per_segment_raw.reshape(self.n_channels, -1)
                elif features_per_segment_raw.ndim == 2 and features_per_segment_raw.shape[0] == self.n_channels: # Already [C,F]
                    node_x_features_shaped = features_per_segment_raw
                else: # Unhandled shape
                    print(f"     ⚠️ Error: Feature set {feature_set_idx} has unexpected shape {features_per_segment_raw.shape}. Expected [C*F] or [C,F]. Skipping.")
                    continue
            except ValueError as e:
                print(f"     ⚠️ Error reshaping features for segment {feature_set_idx}: {e}. Original shape: {features_per_segment_raw.shape}. Skipping.")
                continue
            
            node_x_features = node_x_features_shaped.copy()

            if self.apply_normalization:
                # Normalize each channel's feature vector independently
                mean = node_x_features.mean(axis=1, keepdims=True)
                std = node_x_features.std(axis=1, keepdims=True) + 1e-6
                node_x_features = (node_x_features - mean) / std

            x_tensor = torch.tensor(node_x_features, dtype=torch.float) # Shape [C, F_per_channel]
            edge_index = self._create_edges(node_x_features) # Pass [C, F_per_channel] data

            if edge_index.numel() == 0 and edge_index.shape[0] != 2:
                print(f"     ⚠️ No edges created for feature item {feature_set_idx}. Skipping.")
                continue

            y_tensor = None
            if has_labels_in_clips_df:
                label_val = self.clips_df[ClipsDF.label].iloc[feature_set_idx]
                if pd.notna(label_val):
                    y_tensor = torch.tensor([label_val], dtype=torch.float)
            
            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, original_idx=torch.tensor([feature_set_idx]))
            save_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{save_idx}.pt')
            torch.save(data, save_path)
            save_idx += 1
        
        self._processed_items_count = save_idx
        print(f"   ✅ Processed and saved {self._processed_items_count} items from features.")

    def _process_from_sessions(self):
        save_idx = 0
        has_labels = ClipsDF.label in self.clips_df.columns
        if not has_labels:
            print(f"   ⚠️ Warning: '{ClipsDF.label}' column not found in clips_df. Labels will be None for segment processing.")

        num_total_segments = len(self.clips_df)
        print(f"   - Total segments to process from clips_df: {num_total_segments}")
        log_interval = max(1, num_total_segments // 10)

        for row_idx, (_, row_data) in enumerate(self.clips_df.iterrows()):
            if (row_idx + 1) % log_interval == 0 or row_idx == num_total_segments -1 :
                print(f"     - Processing segment {row_idx + 1}/{num_total_segments} (Original index: {row_data.name if hasattr(row_data, 'name') else 'N/A'})...")
            
            signal_file_name = row_data[ClipsDF.signals_path]
            # Assume signal_folder is the base directory for these files
            full_signal_path = self.signal_folder / signal_file_name
            
            if not full_signal_path.exists(): # Try path as is, if not found in signal_folder
                full_signal_path = Path(signal_file_name)
                if not full_signal_path.exists():
                    print(f"     ⚠️ Warning: Signal file {signal_file_name} (tried in {self.signal_folder} and as direct path) not found. Skipping segment {row_idx}.")
                    continue
            
            try:
                session_signal_df = pd.read_parquet(full_signal_path)
                if not all(ch in session_signal_df.columns for ch in self.channels):
                    print(f"     ⚠️ Warning: Not all expected channels {self.channels} found in {full_signal_path}. Available: {session_signal_df.columns.tolist()}. Skipping segment.")
                    continue
                # Ensure channel order matches self.channels
                session_signal_values = session_signal_df[self.channels].values # Shape [Time, Channels]
                
                start_idx = int(row_data[ClipsDF.start_time] * self.sampling_rate)
                end_idx = int(row_data[ClipsDF.end_time] * self.sampling_rate)

                if end_idx > session_signal_values.shape[0]:
                    print(f"     ⚠️ Warning: Segment {row_idx} end_idx {end_idx} exceeds signal length {session_signal_values.shape[0]}. Clamping.")
                    end_idx = session_signal_values.shape[0]
                if start_idx >= end_idx :
                    print(f"     ⚠️ Warning: Segment {row_idx} start_idx {start_idx} >= end_idx {end_idx}. Skipping.")
                    continue
                segment_signal_values = session_signal_values[start_idx:end_idx, :] # Shape [Segment_Time, Channels]
            except Exception as e:
                print(f"     ⚠️ Error loading/slicing signal for segment {row_idx} from {full_signal_path}: {e}. Skipping.")
                continue

            if segment_signal_values.shape[0] == 0:
                print(f"     ⚠️ Segment {row_idx} is empty after slicing. Skipping.")
                continue

            # Preprocess: expects [Time, Channels], returns [Time, Channels]
            processed_signal_time_channels = self._preprocess_signal(segment_signal_values) 
            
            # For PyG Data object, x is typically [Channels, Features_per_node]
            # Here, features_per_node is the time series for that channel. So, [Channels, Segment_Time]
            x_tensor_data = processed_signal_time_channels.T # Transpose to [Channels, Segment_Time]
            x_tensor = torch.tensor(x_tensor_data, dtype=torch.float)

            # Create edges based on the [Channels, Segment_Time] data
            edge_index = self._create_edges(x_tensor_data)

            if edge_index.numel() == 0 and edge_index.shape[0] != 2:
                print(f"     ⚠️ No edges created for segment {row_idx}. Skipping.")
                continue

            y_tensor = None
            if has_labels and ClipsDF.label in row_data.index and pd.notna(row_data[ClipsDF.label]):
                y_tensor = torch.tensor([row_data[ClipsDF.label]], dtype=torch.float)
            
            data_obj_attrs = {
                'x': x_tensor, 'edge_index': edge_index, 'y': y_tensor,
                'original_clips_df_idx': torch.tensor([row_idx]) # Store original index from clips_df
            }
            # Add other metadata if needed, e.g., patient_id from ClipsDF constants
            if ClipsDF.patient in row_data.index and pd.notna(row_data[ClipsDF.patient]):
                 data_obj_attrs['patient_id_str'] = str(row_data[ClipsDF.patient]) # Ensure it's a string if used later for identification
            
            data = Data(**data_obj_attrs)
            save_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{save_idx}.pt')
            torch.save(data, save_path)
            save_idx += 1
            
        self._processed_items_count = save_idx
        print(f"   ✅ Processed and saved {self._processed_items_count} segments from raw signals.")

    def _preprocess_signal(self, signal_array_time_channels: np.ndarray) -> np.ndarray:
        # Input: signal_array [Time, Channels]
        # Output: processed_signal_array [Time, Channels]
        processed_signal = signal_array_time_channels.copy()
        
        if self.apply_filtering and self.bp_filter_coeffs is not None:
            processed_signal = signal.sosfiltfilt(self.bp_filter_coeffs, processed_signal, axis=0)
            if self.notch_filter_coeffs is not None:
                processed_signal = signal.sosfiltfilt(self.notch_filter_coeffs, processed_signal, axis=0)
        
        if self.apply_rereferencing:
            # Assumes rereference function expects [Time, Channels]
            processed_signal = rereference(processed_signal) # Uses the imported or defined rereference

        if self.apply_normalization:
            # Normalize each channel (column) independently over time
            mean = np.mean(processed_signal, axis=0, keepdims=True)
            std = np.std(processed_signal, axis=0, keepdims=True) + 1e-6
            processed_signal = (processed_signal - mean) / std
            
        return processed_signal

    def _load_spatial_distances(self) -> Dict[Tuple[str, str], float]:
        print("   - Loading spatial distances...")
        spatial_distances_map = {}
        try:
            df_distances = pd.read_csv(self.spatial_distance_file)
            for _, row in df_distances.iterrows():
                ch1, ch2, dist = row["from"], row["to"], row["distance"]
                # Only include channels defined in self.channels
                if ch1 not in self.channels or ch2 not in self.channels:
                    continue
                spatial_distances_map[(ch1, ch2)] = float(dist)
                spatial_distances_map[(ch2, ch1)] = float(dist) # Ensure symmetry
            print(f"     - Loaded {len(spatial_distances_map)//2} unique spatial distances relevant to defined channels.")
        except Exception as e:
            print(f"     ⚠️ Error loading spatial distances from {self.spatial_distance_file}: {e}")
            raise
        return spatial_distances_map

    def _create_edges(self, node_feature_data: np.ndarray) -> torch.Tensor:
        # node_feature_data is expected to be [Channels, Features_per_channel (e.g., Time or D_embed or F_feat)]
        if self.edge_strategy == "spatial":
            if self.spatial_distances_map is None:
                 raise ValueError("Spatial distances not loaded for 'spatial' edge strategy.")
            return self._create_spatial_edges()
        elif self.edge_strategy == "correlation":
            # For correlation, data should be appropriate (e.g., time series or comparable features)
            if node_feature_data.shape[1] <= 1 and self.mode != "embedding" and self.mode != "feature": # If it's time series and too short
                print("     ⚠️ Warning: Correlation edge creation skipped, data per channel too short (<=1 timepoint/feature).")
                return torch.empty((2,0), dtype=torch.long)
            return self._create_correlation_edges(node_feature_data)
        elif self.edge_strategy == "full": 
            return self._create_fully_connected_edges()
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")

    def _create_spatial_edges(self) -> torch.Tensor:
        edge_list = []
        # Create a mapping from channel name to its index in self.channels
        channel_to_idx = {name: i for i, name in enumerate(self.channels)}
        
        for i, ch1_name in enumerate(self.channels):
            for j, ch2_name in enumerate(self.channels):
                if i < j: # Avoid self-loops and duplicate pairs (e.g. (A,B) and (B,A) initially)
                    # Check if the pair exists in the loaded spatial distances
                    if (ch1_name, ch2_name) in self.spatial_distances_map:
                        # Add edge in both directions for undirected graph
                        edge_list.append([channel_to_idx[ch1_name], channel_to_idx[ch2_name]])
                        edge_list.append([channel_to_idx[ch2_name], channel_to_idx[ch1_name]])
        
        if not edge_list: return torch.empty((2,0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _create_correlation_edges(self, channel_data: np.ndarray) -> torch.Tensor:
        # channel_data is [Channels, Features_per_channel]
        # Correlation is typically computed row-wise if rows are variables (channels)
        # and columns are observations (time points/features).
        # np.corrcoef(channel_data) will compute correlation between rows (channels).
        
        if channel_data.shape[1] <= 1: # Not enough features/timepoints for meaningful correlation
            return torch.empty((2,0), dtype=torch.long)
        
        try:
            # Pearson correlation between channel features/time-series
            corr_matrix = np.corrcoef(channel_data) 
            # Handle potential NaNs (e.g. if a channel has zero variance)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr_matrix, 0) # No self-loops from correlation
            abs_corr_matrix = np.abs(corr_matrix)
        except Exception as e:
            print(f"     ⚠️ Error computing correlation matrix: {e}. Returning empty edges.")
            return torch.empty((2,0), dtype=torch.long)

        edge_list = []
        num_nodes = channel_data.shape[0] # Should be self.n_channels

        if self.top_k_correlation is not None and self.top_k_correlation > 0:
            # For each node, find its top_k connections if they meet the threshold (if one is also applied)
            temp_edges = set() # Use a set of sorted tuples to ensure unique undirected edges
            for i in range(num_nodes):
                # Sort correlations for node i in descending order
                # Exclude self-correlation by looking at indices j != i
                sorted_neighbor_indices = np.argsort(abs_corr_matrix[i, :])[::-1]
                
                added_for_node_i = 0
                for j in sorted_neighbor_indices:
                    if i == j: continue # Skip self
                    if added_for_node_i >= self.top_k_correlation: break # Reached top_k for this node

                    if abs_corr_matrix[i, j] >= self.correlation_threshold:
                        # Add sorted tuple to set to handle (i,j) and (j,i) as one edge
                        temp_edges.add(tuple(sorted((i, j))))
                        added_for_node_i += 1
            
            # Create symmetric edges from unique pairs
            for u, v in sorted(list(temp_edges)): # Sort for reproducibility
                edge_list.append([u,v])
                edge_list.append([v,u])
        else: # Threshold-based connections
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes): # Iterate over upper triangle
                    if abs_corr_matrix[i, j] >= self.correlation_threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i]) # Add symmetric edge
                        
        if not edge_list: return torch.empty((2,0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _create_fully_connected_edges(self) -> torch.Tensor:
        """
        Creates a fully connected undirected graph of electrodes.

        Returns:
            torch.Tensor: Edge index tensor
        """
        num_nodes = len(self.channels)
        edge_list = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])  # Include edge in both directions

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index


    def len(self) -> int:
        return self._processed_items_count

    def get(self, idx: int) -> Data:
        if not (0 <= idx < self._processed_items_count):
             raise IndexError(f"Index {idx} is out of bounds for dataset of length {self._processed_items_count}.")

        if hasattr(self, '_data_list') and self._data_list is not None: # Prefetched
            if idx < len(self._data_list) and self._data_list[idx] is not None:
                return self._data_list[idx]
            # If prefetch list is somehow misaligned or item is None, fall through to load from disk.
            # This might happen if prefetch was partial or failed for some items.

        file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{idx}.pt')
        try:
            # weights_only=False is important for loading Data objects with arbitrary Python objects (though less common now)
            # For PyTorch 2.1+, torch.load has stricter unpickling by default.
            # If Data object only contains tensors, strings, numbers, it might be fine.
            # But to be safe for general Data objects:
            data = torch.load(file_path) # PyTorch 2.1+ might need weights_only=False if there are non-tensor attributes being pickled
            # For PyG Data objects, usually they are safe, but if custom attributes are added:
            # data = torch.load(file_path, weights_only=False) # More robust for older PyTorch or complex Data objects
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed file {file_path} not found for index {idx}. "
                                    "Ensure dataset is processed or check path.")
        except Exception as e:
            # More specific error handling might be needed based on torch version and typical errors
            if "weights_only" in str(e).lower(): # Heuristic for common PyTorch 2.1+ error with old pickles
                 print(f"     INFO: Attempting torch.load with weights_only=False for {file_path}")
                 data = torch.load(file_path, weights_only=False) # Retry
            else:
                 raise RuntimeError(f"Error loading processed file {file_path} for index {idx}: {e}")
        return data
    
    def _prefetch_data(self):
        num_items_to_prefetch = self.len()
        if num_items_to_prefetch == 0:
            print("   - Prefetch: No items to prefetch.")
            self._data_list = []
            return

        self._data_list = [None] * num_items_to_prefetch
        print(f"🚀 Starting prefetch of {num_items_to_prefetch} items for {self.mode} mode...")
        prefetched_count = 0
        log_interval = max(1, num_items_to_prefetch // 10)

        for i in range(num_items_to_prefetch):
            if (i + 1) % log_interval == 0 or i == num_items_to_prefetch -1 :
                print(f"   - Prefetching item {i + 1}/{num_items_to_prefetch}...")
            try:
                # Use the get method to ensure consistent loading logic, but this might be recursive if get itself checks _data_list.
                # Direct loading is better here.
                file_path = osp.join(self.processed_dir, f'{self._current_file_prefix}{i}.pt')
                if osp.exists(file_path):
                    try:
                        self._data_list[i] = torch.load(file_path)
                    except Exception as e_load: # Catch PyTorch 2.1+ common error
                         if "weights_only" in str(e_load).lower():
                            self._data_list[i] = torch.load(file_path, weights_only=False)
                         else:
                            raise # Re-raise other errors
                    prefetched_count +=1
                else:
                    print(f"   ⚠️ Warning: File {file_path} not found during prefetch for index {i}.")
            except Exception as e:
                print(f"   ⚠️ Error prefetching item {i} from {file_path}: {e}")
        
        # Filter out Nones if any files were missing or failed to load, though ideally all should load.
        loaded_items = [item for item in self._data_list if item is not None]
        if len(loaded_items) != prefetched_count:
             print(f"   ⚠️ Discrepancy in prefetch counts. Successfully loaded: {len(loaded_items)} out of attempted {prefetched_count}")
        
        self._data_list = loaded_items # Store only successfully loaded items
        if len(self._data_list) != num_items_to_prefetch:
             print(f"   ⚠️ Warning: Not all items were successfully prefetched. Expected {num_items_to_prefetch}, loaded {len(self._data_list)}")


        print(f"🏁 Prefetching complete. Loaded {len(self._data_list)} items into memory.")
