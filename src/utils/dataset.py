# --------------------------------------- General imports ---------------------------------------#
import os.path as osp
import os
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Callable

import pandas as pd
import numpy as np
import networkx as nx
from scipy import signal

import torch
from torch_geometric.data import Dataset, Data
from src.utils.signal import time_filtering, rereference, normalize

class ClipsDF:
    # Index names if using ensure_eeg_multiindex
    patient = 'patient'
    session = 'session'
    clip_idx = 'clip' 
    segment = 'segment'

    # Column names expected in the input 'clips' DataFrame
    start_time = 'start_time'
    end_time = 'end_time'
    label = 'label' 
    signals_path = 'signals_path' 
    sampling_rate = 'sampling_rate'

class GraphEEGDataset(Dataset):
    def __init__(
        self,
        root: str,
        clips_df: pd.DataFrame,
        signal_folder: str,
        extracted_features: Optional[np.ndarray] = None,
        use_extracted_features: bool = True,
        edge_strategy: str = "spatial",
        spatial_distance_file: Optional[str] = None,
        correlation_threshold: float = 0.7,
        top_k_correlation: Optional[int] = None,
        force_reprocess: bool = False,
        bandpass_frequencies: Tuple[float, float] = (0.5, 50.0),
        notch_freq_hz: Optional[float] = 60.0,
        notch_q_factor: float = 30.0,
        apply_filtering: bool = True,
        apply_rereferencing: bool = True,
        apply_normalization: bool = True,
        prefetch_data: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        print(f"🚀 Initializing GraphEEGDataset at root: {root}")
        self.clips_df = clips_df
        self.signal_folder = Path(signal_folder)
        
        self.use_extracted_features_mode = use_extracted_features and (extracted_features is not None)
        self.node_features_source = extracted_features if self.use_extracted_features_mode else None
        print(f"   - Mode: {'Using pre-extracted features' if self.use_extracted_features_mode else 'Processing raw signals'}")

        self.edge_strategy = edge_strategy
        self.spatial_distance_file = spatial_distance_file
        self.correlation_threshold = correlation_threshold
        self.top_k_correlation = top_k_correlation
        print(f"   - Edge strategy: {self.edge_strategy}")
        
        self.force_reprocess_flag = force_reprocess

        self.apply_filtering = apply_filtering
        self.apply_rereferencing = apply_rereferencing
        self.apply_normalization = apply_normalization
        self.bandpass_frequencies = bandpass_frequencies
        self.notch_freq_hz = notch_freq_hz
        self.notch_q_factor = notch_q_factor
        print(f"   - Preprocessing: Filtering={apply_filtering}, Rereferencing={apply_rereferencing}, Normalization={apply_normalization}")

        if not isinstance(clips_df, pd.DataFrame):
            raise TypeError("'clips_df' must be a pandas DataFrame.")
        
        if ClipsDF.sampling_rate not in clips_df.columns:
            raise ValueError(f"Column '{ClipsDF.sampling_rate}' not found in clips_df.")
        unique_sampling_rates = clips_df[ClipsDF.sampling_rate].unique()
        if len(unique_sampling_rates) > 1:
            raise ValueError("Multiple sampling rates found in clips_df. Dataset expects a single sampling rate.")
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

        self.channels = ["FP1","FP2","F3","F4","C3","C4","P3","P4","O1","O2",
                           "F7","F8","T3","T4","T5","T6","FZ","CZ","PZ"]
        self.n_channels = len(self.channels)

        if self.apply_filtering:
            print(f"   - Designing filters: Bandpass {self.bandpass_frequencies} Hz, Notch {self.notch_freq_hz} Hz")
            self.bp_filter_coeffs = signal.butter(4, self.bandpass_frequencies, btype="bandpass",
                                                  output="sos", fs=self.sampling_rate)
            if self.notch_freq_hz is not None:
                notch_coeffs_ba = signal.iirnotch(w0=self.notch_freq_hz, Q=self.notch_q_factor, fs=self.sampling_rate)
                self.notch_filter_coeffs = signal.tf2sos(*notch_coeffs_ba)
            else:
                self.notch_filter_coeffs = None
        
        self.spatial_distances_map = None
        if self.edge_strategy == "spatial":
            if self.spatial_distance_file is None:
                raise ValueError("spatial_distance_file must be provided for 'spatial' edge strategy.")
            self.spatial_distances_map = self._load_spatial_distances()

        self._data_list = None 
        
        print("   - Calling PyG Dataset super().__init__()...")
        super().__init__(root, transform, pre_transform, pre_filter)
        print("   - PyG Dataset super().__init__() call complete.")

        if prefetch_data:
            self._prefetch_data()
        print(f"🏁 GraphEEGDataset initialization complete. Found {self.len()} items.")

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.use_extracted_features_mode:
            return ["extracted_features_placeholder.npy"] 
        
        if ClipsDF.signals_path in self.clips_df.columns:
            unique_signal_files = self.clips_df[ClipsDF.signals_path].unique().tolist()
            return [str(Path(f).name) for f in unique_signal_files] 
        return []

    @property
    def processed_file_names(self) -> List[str]:
        if self.use_extracted_features_mode:
            if self.node_features_source is None: return []
            return [f'data_feat_{i}.pt' for i in range(len(self.node_features_source))]
        else:
            return [f'data_segment_{i}.pt' for i in range(len(self.clips_df))]

    @property
    def labels(self) -> Optional[pd.Series]:
        """
        Returns a Series of labels if available in clips_df, otherwise None.
        This is useful for classification tasks.
        """
        if ClipsDF.label in self.clips_df.columns:
            return self.clips_df[ClipsDF.label]
        return None

    def process(self):
        print(f"⚙️ process() called. Target processed directory: {self.processed_dir}")
        if self.force_reprocess_flag:
            print(f"   - Force reprocess: Deleting existing files in {self.processed_dir}...")
            deleted_count = 0
            if osp.exists(self.processed_dir):
                for f_name in os.listdir(self.processed_dir):
                    if f_name.startswith('data_') and f_name.endswith('.pt'):
                        os.remove(osp.join(self.processed_dir, f_name))
                        deleted_count +=1
            print(f"     - Deleted {deleted_count} existing .pt files.")
        
        if not osp.exists(self.processed_dir): # Ensure processed_dir exists
            print(f"   - Creating processed directory: {self.processed_dir}")
            os.makedirs(self.processed_dir)

        if self.use_extracted_features_mode:
            print("   - Starting processing from pre-extracted features...")
            self._process_from_features()
        else:
            print("   - Starting processing from raw signal sessions...")
            self._process_from_sessions()
        print("🏁 process() finished.")

    def _process_from_features(self):
        processed_idx = 0
        has_labels = ClipsDF.label in self.clips_df.columns
        if not has_labels:
            print(f"   ⚠️ Warning: '{ClipsDF.label}' column not found in clips_df. Labels will be None for feature processing.")

        num_total_features = len(self.node_features_source)
        print(f"   - Total features to process: {num_total_features}")
        log_interval = max(1, num_total_features // 10) # Log every 10% or at least 1

        if len(self.node_features_source) != len(self.clips_df) and has_labels:
            print(f"   ⚠️ Warning: Length of extracted_features ({len(self.node_features_source)}) "
                  f"does not match length of clips_df ({len(self.clips_df)}). "
                  "Label assignment might be incorrect if labels are present.")

        for feature_idx, features_per_segment in enumerate(self.node_features_source):
            if (feature_idx + 1) % log_interval == 0 or feature_idx == num_total_features -1 :
                print(f"     - Processing feature item {feature_idx + 1}/{num_total_features}...")
            try:
                node_x_features = features_per_segment.reshape(self.n_channels, -1)
            except ValueError as e:
                print(f"     ⚠️ Error reshaping features for segment {feature_idx}: {e}. Skipping.")
                continue
            
            if self.apply_normalization:
                mean = node_x_features.mean(axis=1, keepdims=True)
                std = node_x_features.std(axis=1, keepdims=True) + 1e-6
                node_x_features = (node_x_features - mean) / std

            x_tensor = torch.tensor(node_x_features, dtype=torch.float)
            edge_index = self._create_edges(node_x_features)

            if edge_index.numel() == 0 and edge_index.shape[0] != 2:
                continue

            y_tensor = None
            if has_labels and feature_idx < len(self.clips_df):
                label_val = self.clips_df[ClipsDF.label].iloc[feature_idx]
                if pd.notna(label_val):
                    y_tensor = torch.tensor([label_val], dtype=torch.float)
            
            data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, segment_id=torch.tensor([feature_idx]))
            save_path = osp.join(self.processed_dir, f'data_feat_{processed_idx}.pt')
            torch.save(data, save_path)
            # print(f"       - Saved: {save_path}") # Can be too verbose
            processed_idx += 1
        print(f"   ✅ Processed and saved {processed_idx} items from features.")

    def _process_from_sessions(self):
        processed_segment_count = 0
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
            full_signal_path = self.signal_folder / signal_file_name
            
            if not full_signal_path.exists():
                full_signal_path = Path(signal_file_name) # Try as absolute/relative
                if not full_signal_path.exists():
                    print(f"     ⚠️ Warning: Signal file {signal_file_name} not found. Skipping segment {row_idx}.")
                    continue
            
            try:
                # print(f"       - Loading signal from: {full_signal_path}") # Can be verbose
                session_signal_df = pd.read_parquet(full_signal_path)
                # Ensure columns match self.channels, or select them
                if not all(ch in session_signal_df.columns for ch in self.channels):
                    print(f"     ⚠️ Warning: Not all expected channels {self.channels} found in {full_signal_path}. Available: {session_signal_df.columns.tolist()}. Skipping segment.")
                    continue
                session_signal_values = session_signal_df[self.channels].values
                
                start_idx = int(row_data[ClipsDF.start_time] * self.sampling_rate)
                end_idx = int(row_data[ClipsDF.end_time] * self.sampling_rate)

                if end_idx > session_signal_values.shape[0]:
                    end_idx = session_signal_values.shape[0]
                if start_idx >= end_idx :
                    print(f"     ⚠️ Warning: Segment {row_idx} start_idx {start_idx} >= end_idx {end_idx}. Skipping.")
                    continue
                segment_signal_values = session_signal_values[start_idx:end_idx, :]
            except Exception as e:
                print(f"     ⚠️ Error loading/slicing signal for segment {row_idx} from {full_signal_path}: {e}. Skipping.")
                continue

            if segment_signal_values.shape[0] == 0:
                continue

            # print(f"       - Preprocessing signal for segment {row_idx}...") # Can be verbose
            processed_signal = self._preprocess_signal(segment_signal_values)
            
            x_tensor_data = processed_signal.T 
            x_tensor = torch.tensor(x_tensor_data, dtype=torch.float)

            edge_index = self._create_edges(x_tensor_data)

            if edge_index.numel() == 0 and edge_index.shape[0] != 2:
                continue

            y_tensor = None
            if has_labels and ClipsDF.label in row_data.index and pd.notna(row_data[ClipsDF.label]):
                y_tensor = torch.tensor([row_data[ClipsDF.label]], dtype=torch.float)
            
            data_obj_attrs = {
                'x': x_tensor, 'edge_index': edge_index, 'y': y_tensor,
                'original_clips_df_idx': torch.tensor([row_idx]) 
            }
            if ClipsDF.patient in row_data.index:
                 data_obj_attrs['patient_id_str'] = str(row_data[ClipsDF.patient])
            
            data = Data(**data_obj_attrs)
            save_path = osp.join(self.processed_dir, f'data_segment_{processed_segment_count}.pt')
            torch.save(data, save_path)
            # print(f"       - Saved: {save_path}") # Can be too verbose
            processed_segment_count += 1
        print(f"   ✅ Processed and saved {processed_segment_count} segments from raw signals.")

    def _preprocess_signal(self, signal_array: np.ndarray) -> np.ndarray:
        processed_signal = signal_array.copy()
        # print("     Applying _preprocess_signal:") # Can be verbose
        if self.apply_filtering:
            # print("       - Applying time filtering...") # Can be verbose
            processed_signal = signal.sosfiltfilt(self.bp_filter_coeffs, processed_signal, axis=0)
            if self.notch_filter_coeffs is not None:
                processed_signal = signal.sosfiltfilt(self.notch_filter_coeffs, processed_signal, axis=0)
        if self.apply_rereferencing:
            # print("       - Applying rereferencing...") # Can be verbose
            processed_signal = rereference(processed_signal)
        if self.apply_normalization:
            # print("       - Applying normalization...") # Can be verbose
            processed_signal = normalize(processed_signal)
        return processed_signal

    def _load_spatial_distances(self) -> Dict[Tuple[str, str], float]:
        print("   - Loading spatial distances...")
        spatial_distances_map = {}
        try:
            df_distances = pd.read_csv(self.spatial_distance_file)
            for _, row in df_distances.iterrows():
                ch1, ch2, dist = row["from"], row["to"], row["distance"]
                if ch1 not in self.channels or ch2 not in self.channels:
                    continue
                spatial_distances_map[(ch1, ch2)] = dist
                spatial_distances_map[(ch2, ch1)] = dist
            print(f"     - Loaded {len(spatial_distances_map)//2} unique spatial distances.")
        except Exception as e:
            print(f"     ⚠️ Error loading spatial distances from {self.spatial_distance_file}: {e}")
            raise # Re-raise error as this is critical if strategy is spatial
        return spatial_distances_map

    def _create_edges(self, node_feature_data: np.ndarray) -> torch.Tensor:
        # print(f"     Creating edges using strategy: {self.edge_strategy}") # Can be verbose
        if self.edge_strategy == "spatial":
            if self.spatial_distances_map is None:
                 raise ValueError("Spatial distances not loaded for 'spatial' edge strategy.")
            return self._create_spatial_edges()
        elif self.edge_strategy == "correlation":
            return self._create_correlation_edges(node_feature_data)
        elif self.edge_strategy == "full": 
            return self._create_fully_connected_edges()
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")

    def _create_spatial_edges(self) -> torch.Tensor:
        edge_list = []
        channel_to_idx = {name: i for i, name in enumerate(self.channels)}
        for i, ch1_name in enumerate(self.channels):
            for j, ch2_name in enumerate(self.channels):
                if i < j: 
                    if (ch1_name, ch2_name) in self.spatial_distances_map:
                        edge_list.append([channel_to_idx[ch1_name], channel_to_idx[ch2_name]])
                        edge_list.append([channel_to_idx[ch2_name], channel_to_idx[ch1_name]])
        # print(f"       - Created {len(edge_list)//2} spatial edges.") # Can be verbose
        if not edge_list: return torch.empty((2,0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _create_correlation_edges(self, channel_data: np.ndarray) -> torch.Tensor:
        if channel_data.shape[1] <= 1: 
            return torch.empty((2,0), dtype=torch.long) 
        try:
            corr_matrix = np.corrcoef(channel_data)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(corr_matrix, 0) 
            abs_corr_matrix = np.abs(corr_matrix)
        except Exception:
            return torch.empty((2,0), dtype=torch.long)

        edge_list = []
        num_nodes = channel_data.shape[0]

        if self.top_k_correlation is not None and self.top_k_correlation > 0:
            # Simplified top_k: for each node, find its top_k connections if they meet threshold
            # This can result in more than top_k * num_nodes / 2 total edges if connections are mutual.
            # A more global top_k would require sorting all possible edges.
            temp_edges = []
            for i in range(num_nodes):
                row_corrs = abs_corr_matrix[i, :]
                # Get indices of top_k largest correlations (excluding self, which is 0)
                # Argsort sorts smallest to largest, so take last k after excluding self.
                # Need to be careful if less than k other nodes exist.
                potential_neighbors = np.argsort(row_corrs)[::-1] # Sort descending
                
                added_for_node_i = 0
                for j in potential_neighbors:
                    if i == j: continue
                    if added_for_node_i >= self.top_k_correlation: break
                    if row_corrs[j] >= self.correlation_threshold:
                        temp_edges.append(tuple(sorted((i, j)))) # Store as sorted tuple to handle duplicates
                        added_for_node_i += 1
            
            # Create symmetric edges from unique pairs
            unique_pairs = sorted(list(set(temp_edges))) # Ensure order for reproducibility if needed
            for u, v in unique_pairs:
                edge_list.append([u,v])
                edge_list.append([v,u])
        else: 
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes): 
                    if abs_corr_matrix[i, j] >= self.correlation_threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
        # print(f"       - Created {len(edge_list)//2} correlation edges.") # Can be verbose
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
        # PyG's base class uses self.processed_file_names to determine length if this method isn't overridden
        # or if _data_list isn't used. So, ensure processed_file_names is accurate.
        # For clarity, explicitly define length based on the input source.
        if self.use_extracted_features_mode:
            return len(self.node_features_source) if self.node_features_source is not None else 0
        return len(self.clips_df)

    def get(self, idx: int) -> Data:
        if hasattr(self, '_data_list') and self._data_list is not None and idx < len(self._data_list) and self._data_list[idx] is not None:
             return self._data_list[idx]
        
        file_prefix = 'data_feat_' if self.use_extracted_features_mode else 'data_segment_'
        file_path = osp.join(self.processed_dir, f'{file_prefix}{idx}.pt')
        try:
            # MODIFIED: Added weights_only=False
            data = torch.load(file_path, weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Processed file {file_path} not found. "
                                    "Consider re-running with force_reprocess=True or check data pipeline integrity.")
        except Exception as e:
            raise RuntimeError(f"Error loading processed file {file_path}: {e}")
        return data
    
    def _prefetch_data(self):
        num_items = self.len()
        if num_items == 0:
            print("No items to prefetch.")
            self._data_list = []
            return

        self._data_list = [None] * num_items
        print(f"🚀 Starting prefetch of {num_items} items...")
        prefetched_count = 0
        log_interval = max(1, num_items // 10)

        for i in range(num_items):
            if (i + 1) % log_interval == 0 or i == num_items -1 :
                print(f"   - Prefetching item {i + 1}/{num_items}...")
            try:
                file_prefix = 'data_feat_' if self.use_extracted_features_mode else 'data_segment_'
                file_path = osp.join(self.processed_dir, f'{file_prefix}{i}.pt')
                if osp.exists(file_path):
                    # MODIFIED: Added weights_only=False
                    self._data_list[i] = torch.load(file_path, weights_only=False)
                    prefetched_count +=1
                else:
                    print(f"   ⚠️ Warning: File {file_path} not found during prefetch for index {i}.")
            except Exception as e:
                print(f"   ⚠️ Error prefetching item {i} from {file_path}: {e}")
        
        # Filter out Nones if any files were missing or failed to load
        self._data_list = [item for item in self._data_list if item is not None]
        if len(self._data_list) != prefetched_count : # Should be same if no items were None initially
             print(f"   ⚠️ Discrepancy in prefetch counts. Successfully loaded: {len(self._data_list)}")

        print(f"🏁 Prefetching complete. Loaded {len(self._data_list)} items into memory.")


# --- Utility Function (can be outside the class or in a utils.py) ---
def ensure_eeg_multiindex(df: pd.DataFrame, id_col_name: Optional[str] = 'id') -> pd.DataFrame:
    """
    Ensures the DataFrame has a MultiIndex with levels ['patient', 'session', 'clip', 'segment'].
    ID format expected: <patient_id>_s<session_no>_t<clip_no>_<segment_no>
    """
    df_out = df.copy()
    desired_names = [ClipsDF.patient, ClipsDF.session, ClipsDF.clip_idx, ClipsDF.segment]

    if isinstance(df_out.index, pd.MultiIndex):
        return df_out

    id_series: Optional[pd.Series] = None
    if id_col_name and id_col_name in df_out.columns:
        id_series = df_out[id_col_name]
    elif not isinstance(df_out.index, pd.MultiIndex):
        id_series = pd.Series(df_out.index.values, index=df_out.index)
    else:
        raise ValueError("Cannot determine ID source for MultiIndex creation. "
                         "Provide string IDs in a column or as a simple index.")

    if id_series is None:
        raise ValueError("Could not determine a valid source for string IDs.")

    parsed_ids = []
    for record_id_val in id_series:
        record_id_str = str(record_id_val)
        parts = record_id_str.split('_')
        if len(parts) != 4:
            raise ValueError(f"ID '{record_id_str}' malformed. Expected 4 parts, got {len(parts)}.")
        try:
            patient = parts[0]
            session = int(parts[1][1:]) 
            clip_val = int(parts[2][1:])    
            segment_val = int(parts[3])
            parsed_ids.append((patient, session, clip_val, segment_val))
        except (ValueError, IndexError) as e:
            raise ValueError(f"Error parsing components of ID '{record_id_str}': {e}") from e
    
    if not parsed_ids: 
        df_out.index = pd.MultiIndex(levels=[[]]*len(desired_names), codes=[[]]*len(desired_names), names=desired_names)
        return df_out

    new_multiindex = pd.MultiIndex.from_tuples(parsed_ids, names=desired_names)

    if len(new_multiindex) != len(df_out):
        raise AssertionError("Internal error: Mismatch in length of new MultiIndex and DataFrame rows.")

    df_out.index = new_multiindex
    return df_out
