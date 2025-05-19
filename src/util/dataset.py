import os.path as osp
import os
from typing import Optional, Dict, Tuple, List

import pandas as pd
import numpy as np
import networkx as nx
from scipy import signal

import torch
from torch_geometric.data import Dataset, Data

from src.util.signal import time_filtering

def ensure_eeg_multiindex(df: pd.DataFrame, id_col_name: Optional[str] = 'id') -> pd.DataFrame:
    """
    Ensures the DataFrame has a MultiIndex with levels ['patient', 'session', 'clip', 'segment'].

    If the DataFrame does not have this specific MultiIndex, this function attempts to create it.
    It uses the column specified by `id_col_name` for the string IDs if the column exists.
    If `id_col_name` is None or the column doesn't exist, it tries to use the DataFrame's
    current index if it's a simple (non-MultiIndex) index containing the string IDs.

    The ID format is expected to be: <patient_id>_s<session_no>_t<clip_no>_<segment_no>
    Example: 'pqejgcff_s001_t000_0'

    Args:
        df (pd.DataFrame): The input DataFrame.
        id_col_name (Optional[str]): The name of the column containing the string IDs.
                                     If None, or if the column is not found, the function
                                     will try to use the existing simple index.

    Returns:
        pd.DataFrame: A new DataFrame with the ensured MultiIndex.
                      The original DataFrame is not modified.

    Raises:
        ValueError: If a valid source for IDs cannot be determined or if IDs are malformed.
    """
    df_out = df.copy()  # Work on a copy
    desired_names = ['patient', 'session_clip', 'segment']

    # Check if the DataFrame already has the desired MultiIndex
    if isinstance(df_out.index, pd.MultiIndex) and list(df_out.index.names) == desired_names:
        return df_out

    id_series: Optional[pd.Series] = None

    # Strategy 1: Use the specified id_col_name if it exists as a column
    if id_col_name and id_col_name in df_out.columns:
        id_series = df_out[id_col_name]
    # Strategy 2: Use the current index if id_col_name is None (or col not found)
    # AND current index is simple (not a MultiIndex)
    elif not isinstance(df_out.index, pd.MultiIndex):
        if id_col_name and df_out.index.name != id_col_name:
            # id_col_name was specified but not found as a column.
            # The current index name also doesn't match.
            # We proceed using the current simple index but warn the user.
            print(f"Warning: ID column '{id_col_name}' not found. "
                  f"Using current simple index (name: '{df_out.index.name}') as ID source.")
        elif df_out.index.name:
            print(f"Using current simple index (name: '{df_out.index.name}') as ID source.")
        else:
            print("Using current unnamed simple index as ID source.")
        # Create a Series from the index, ensuring it aligns with df_out for safety
        id_series = pd.Series(df_out.index.values, index=df_out.index)
    else:
        # This case means:
        # 1. df_out.index IS a MultiIndex, but not the desired one (checked at the start).
        # 2. AND (id_col_name was None OR id_col_name was not in df_out.columns).
        # In this situation, we cannot proceed without a clear source for the original string IDs.
        raise ValueError(
            "DataFrame has a MultiIndex, but it's not the desired one. "
            f"To recreate the MultiIndex, the original string IDs must be available either in a "
            f"column (specify `id_col_name`) or the DataFrame should have a simple index of these IDs."
        )

    if id_series is None:
        # This should ideally be caught by the logic above, but as a safeguard:
        raise ValueError(f"Could not determine a valid source for string IDs. "
                         f"Please check `id_col_name` ('{id_col_name}') and the DataFrame's structure.")

    # Parse the IDs
    patients, sessions, clips, segments = [], [], [], []
    for record_id_val in id_series:
        record_id_str = str(record_id_val)  # Ensure it's a string
        parts = record_id_str.split('_')

        if len(parts) != 4:
            raise ValueError(
                f"ID '{record_id_str}' is not in the expected format "
                "<patient_id>_s<session_no>_t<clip_no>_<segment_no>. "
                f"Found {len(parts)} parts: {parts}"
            )
        try:
            patients.append(parts[0])
            sessions.append(int(parts[1][1:]))  # Remove 's' and convert to int
            clips.append(int(parts[2][1:]))     # Remove 't' and convert to int
            segments.append(int(parts[3]))
        except (ValueError, IndexError) as e:
            # ValueError for int conversion, IndexError if parts[x] is too short (e.g. 's' missing)
            raise ValueError(f"Error parsing components of ID '{record_id_str}': {e}") from e

    # Create the new MultiIndex
    new_multiindex = pd.MultiIndex.from_arrays(
        [patients, sessions, clips, segments],
        names=desired_names
    )

    # Ensure the new index has the same length as the DataFrame
    if len(new_multiindex) != len(df_out):
        # This is an internal consistency check and should not be triggered if id_series is derived correctly.
        raise AssertionError(
            f"Internal error: Mismatch between the length of the new MultiIndex ({len(new_multiindex)}) "
            f"and the DataFrame's number of rows ({len(df_out)})."
        )

    df_out.index = new_multiindex
    return df_out

class GraphEEGDataset(Dataset):
    def __init__(
        self,
        root: str,
        clips: pd.DataFrame,
        signal_folder: str,
        extracted_features: np.ndarray,
        selected_features_train: bool,
        edge_strategy: str = "spatial",
        spatial_distance_file: Optional[str] = None,
        correlation_threshold: float = 0.7,
        force_reprocess: bool = False,
        segment_length: int = 12 * 250,  # 12 seconds * 250 Hz -> 3000 samples
        apply_filtering: bool = True,
        apply_rereferencing: bool = True,
        apply_normalization: bool = True,
        sampling_rate: int = 250,
    ):
        """
        Custom PyTorch Geometric dataset for EEG data.

        Args:
            root: Root directory where the dataset should be saved
            clips: DataFrame containing segment information (patient, session, start_time, end_time, label, signal_path)
            signal_folder: Path to the folder containing EEG signal files
            edge_strategy: Strategy to create edges ('spatial' or 'correlation')
            spatial_distance_file: Path to file containing spatial distances (required if edge_strategy='spatial')
            correlation_threshold: Threshold for creating edges based on correlation (used if edge_strategy='correlation')
            transform: Transform to be applied to each data object
            pre_transform: Pre-transform to be applied to each data object
            pre_filter: Pre-filter to be applied to each data object
            sampling_rate: Sampling rate of the EEG data - fixed to 250 Hz
        """
        self.root = root
        self.clips = clips
        self.signal_folder = signal_folder
        self.force_reprocess = force_reprocess
        self.extracted_features = extracted_features
        self.selected_features_train = selected_features_train
        self.edge_strategy = edge_strategy
        self.spatial_distance_file = spatial_distance_file
        self.correlation_threshold = correlation_threshold
        self.segment_length = segment_length
        self.apply_filtering = apply_filtering
        self.apply_rereferencing = apply_rereferencing
        self.apply_normalization = apply_normalization
        self.sampling_rate = sampling_rate

        # EEG channels - standard 10-20 system
        self.channels = [
            "FP1",
            "FP2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "FZ",
            "CZ",
            "PZ",
        ]

        self.n_channels = len(self.channels)  # Number of EEG channels

        # Load spatial distances if applicable
        if edge_strategy == "spatial" and spatial_distance_file is not None:
            self.spatial_distances = self._load_spatial_distances()

        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)

        if self.force_reprocess is True:
            if self.selected_features_train is True:
                self.process_features()
            elif self.selected_features_train is False:
                self.process_sessions()

    def _load_spatial_distances(self) -> Dict:
        """
        Load spatial distances between electrodes.
        Expected format: csv file that can be loaded into a dictionary or matrix
        representing distances between electrodes.

        Returns:
            Dictionary of distances or adjacency matrix
        """
        spatial_distances = (
            {}
        )  # Dictionary to store distances between electrodes with keys (tuple) as (ch1, ch2)
        df_distances = pd.read_csv(self.spatial_distance_file)
        for _, row in df_distances.iterrows():
            ch1 = row["from"]
            ch2 = row["to"]
            dist = row["distance"]

            spatial_distances[(ch1, ch2)] = dist
            spatial_distances[(ch2, ch1)] = dist

        return spatial_distances

    def process_features(self):
        """
        Processes extracted features from samples into PyTorch Geometric Data objects.
        """
        idx = 0
        for index, segment_signal in enumerate(
            self.extracted_features
        ):  # (samples, extracted_features*electrodes)
            segment_signal = segment_signal.reshape(
                self.n_channels, -1
            )  # (channels, features)
            # Normalize features
            mean = segment_signal.mean(axis=0, keepdims=True)
            std = segment_signal.std(axis=0, keepdims=True) + 1e-6  # to avoid div by 0
            segment_signal = (segment_signal - mean) / std
            x = torch.tensor(segment_signal, dtype=torch.float)
            # Creates a tensor -> graph: each node = 1 EEG channel, and its feature = the full time series

            # Create edges based on the selected strategy
            edge_index = self._create_edges(segment_signal)

            if edge_index.shape == torch.Size([0]):
                continue  # Skip if no edges are created - with correlation strategy: happened when all channels are uncorrelated (very rare)

            # Create label tensor
            y = torch.tensor(
                [self.clips["label"].values[index]], dtype=torch.float
            )  # BCELoss expects float labels

            # Create Data object
            data = Data(
                x=x,  # Node features: channels x time points
                edge_index=edge_index,  # Edges between channels
                y=y,  # Label
            )

            # Save processed data
            torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def process_sessions(self):
        """
        Processes raw data into PyTorch Geometric Data objects.
        """
        sessions = list(
            self.clips.groupby(["patient", "session"])
        )  # List of tuples: ((patient_id, session_id), session_df)
        idx = 0
        for (_, _), session_df in sessions:
            # Load signal data (for complete session)
            session_signal = pd.read_parquet(
                f"{self.signal_folder}/{session_df['signals_path'].values[0]}"  # Any 'signal_path' in session_df points to same parquet
            )

            # ------------------------- Preprocess signal data ---------------------------#
            session_signal = self._preprocess_signal(session_signal)

            #  ------------ Extract corresponding segments from signal ---------------------#
            for _, row in session_df.iterrows():  # Each row corresponds to a segment
                start = int(row["start_time"] * self.sampling_rate)
                end = int(row["end_time"] * self.sampling_rate)
                segment_signal = session_signal[
                    start:end
                ].T  # Transpose to get channels as rows: (3000 time points,19 channel)->(19,3000)

                x = torch.tensor(segment_signal, dtype=torch.float)
                # Creates a tensor -> graph: each node = 1 EEG channel, and its feature = the full time series

                # Create edges based on the selected strategy
                edge_index = self._create_edges(segment_signal)

                if edge_index.shape == torch.Size([0]):
                    continue  # Skip if no edges are created - with correlation strategy: happened when all channels are uncorrelated (very rare)

                # Create label tensor
                y = torch.tensor(
                    [self.clips["label"].values[idx]], dtype=torch.float
                )  # BCELoss expects float labels

                # Create Data object
                data = Data(
                    x=x,  # Node features: channels x time points
                    edge_index=edge_index,  # Edges between channels
                    y=y,  # Label
                )

                # Save processed data
                torch.save(data, osp.join(self.processed_dir, f"data_{idx}.pt"))
                idx += 1

    def _filter_signal(self, signal):
        return time_filtering(signal)

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

    def _create_edges(self, signal_data: pd.DataFrame) -> torch.Tensor:
        """
        Creates edges between EEG channels based on the specified strategy.

        Args:
            signal_data: DataFrame containing EEG signals

        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        """
        if self.edge_strategy == "spatial":
            return self._create_spatial_edges()
        elif self.edge_strategy == "correlation":
            return self._create_correlation_edges(signal_data)
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")

    def _create_spatial_edges(self) -> torch.Tensor:
        """
        Creates edges based on spatial distances between electrodes.

        Returns:
            torch.Tensor: Edge index tensor
        """
        # Create a graph
        G = nx.Graph()

        # Add nodes
        for i, channel in enumerate(self.channels):
            G.add_node(i, name=channel)

        # Add edges based on distances
        edge_list = []
        for i, ch1 in enumerate(self.channels):
            for j, ch2 in enumerate(self.channels):
                if i < j:  # Avoid duplicate edges
                    # Get distance if available, otherwise use a default
                    distance = self.spatial_distances.get(
                        (ch1, ch2), 1.0
                    )  # Get keys (tuple) from dict

                    # Add edge if distance is within threshold (you can adjust this logic)
                    # Here we're adding all edges but you might want to threshold
                    G.add_edge(i, j, weight=distance)
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Add in both directions for PyG

        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index

    def _create_correlation_edges(self, signal_data: pd.DataFrame) -> torch.Tensor:
        """
        Creates edges based on correlation between channel signals.

        Args:
            signal_data: DataFrame containing EEG signals

        Returns:
            torch.Tensor: Edge index tensor
        """
        # Calculate correlation matrix
        corr_matrix = np.abs(np.corrcoef(signal_data))
        if np.isnan(corr_matrix).any():
            raise ValueError("Correlation matrix contains NaNs.")
        if np.isinf(corr_matrix).any():
            raise ValueError("Correlation matrix contains infinite values.")

        # Create edges where correlation exceeds threshold
        edge_list = []
        for i in range(len(self.channels)):
            for j in range(len(self.channels)):
                if j != i and corr_matrix[i, j] >= self.correlation_threshold:
                    edge_list.append(
                        [i, j]
                    )  #  must explicitly add both directions for an undirected edge.

        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # [num_edges, 2] → [2, num_edges]; contiguous(): for row-major order
        return edge_index

    def len(self) -> int:
        """
        Returns the number of examples in the dataset (number of graphs saved)
        """
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """
        Returns the data object at index idx.
        """
        data = torch.load(
            osp.join(self.processed_dir, f"data_{idx}.pt")
        )  # Each pt file contains a Data object / graph representing EEG recording
        return data

    @property
    def processed_file_names(
        self,
    ) -> List[
        str
    ]:  #  Filenames that will be use  in the processed directory (if force_reprocess=False)
        """
        Returns the names of all processed files.
        """

        # return [f"data_{i}.pt" for i in range(len(self.clips))]
        return sorted(
            [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith("data_")
                and f.endswith(".pt")  # Guard against other files
            ]
        )