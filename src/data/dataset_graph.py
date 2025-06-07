# --------------------------------------- General imports ---------------------------------------#
from cProfile import label
import os.path as osp
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union
from collections import defaultdict
import logging
import time

import pandas as pd
import numpy as np
from scipy import signal

import torch
from torch_geometric.data import Dataset, Data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------- Custom imports ---------------------------#
from src.utils.preprocessing_funcs import time_filtering
from src.utils.general_funcs import log


class GraphEEGDataset(Dataset):
    def __init__(
        self,
        root: Path,
        clips: pd.DataFrame,
        signal_folder: Path,
        embeddings_dir: Path,
        use_embeddings: bool,
        extracted_features_dir: Path,
        use_selected_features: bool,
        edge_strategy: str = "spatial",
        spatial_distance_file: Optional[Path] = None,
        correlation_threshold: float = 0.7,
        top_k: Optional[int] = None,
        force_reprocess: bool = True,
        bandpass_frequencies: Tuple[float, float] = (0.5, 50),
        segment_length: int = 12 * 250,  # 12 seconds * 250 Hz -> 3000 samples
        apply_filtering: bool = True,
        apply_rereferencing: bool = True,
        apply_normalization: bool = True,
        sampling_rate: int = 250,
        is_test: bool = False,  # New parameter to indicate test mode
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
            is_test (bool): If True, indicates this is a test dataset without labels
        """
        logger.info("Initializing GraphEEGDataset...")
        logger.info(f"Dataset parameters:")
        logger.info(f"  - Root directory: {root}")
        logger.info(f"  - Edge strategy: {edge_strategy}")
        logger.info(f"  - Top-k neighbors: {top_k}")
        logger.info(f"  - Correlation threshold: {correlation_threshold}")
        logger.info(f"  - Force reprocess: {force_reprocess}")
        logger.info(f"  - Bandpass frequencies: {bandpass_frequencies}")
        logger.info(f"  - Segment length: {segment_length}")
        logger.info(f"  - Apply filtering: {apply_filtering}")
        logger.info(f"  - Apply rereferencing: {apply_rereferencing}")
        logger.info(f"  - Apply normalization: {apply_normalization}")
        logger.info(f"  - Sampling rate: {sampling_rate}")
        logger.info(f"  - Test mode: {is_test}")

        self.root = root
        self.clips = clips
        self.signal_folder = signal_folder
        self.force_reprocess = force_reprocess
        self.embeddings_dir = embeddings_dir
        self.embeddings_train = use_embeddings
        self.extracted_features_dir = extracted_features_dir
        self.selected_features_train = use_selected_features
        self.bandpass_frequencies = bandpass_frequencies
        self.edge_strategy = edge_strategy
        self.top_k = top_k
        self.spatial_distance_file = spatial_distance_file
        self.correlation_threshold = correlation_threshold
        self.segment_length = segment_length
        self.apply_filtering = apply_filtering
        self.apply_rereferencing = apply_rereferencing
        self.apply_normalization = apply_normalization
        self.sampling_rate = sampling_rate
        self.is_test = is_test

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
        logger.info(f"Number of EEG channels: {self.n_channels}")

        # Ids to eliminate
        self.ids_to_eliminate = []

        # Define frequency filters
        logger.info("Setting up signal filters...")
        self.bp_filter = signal.butter(
            4,
            self.bandpass_frequencies,
            btype="bandpass",
            output="sos",
            fs=sampling_rate,
        )
        notch_filter = signal.iirnotch(
            w0=60, Q=30, fs=sampling_rate
        )  # Filter to remove 60 Hz noise (fixed frequency)
        self.notch_filter = signal.tf2sos(*notch_filter)

        # Load spatial distances if applicable
        if edge_strategy == "spatial" and spatial_distance_file is not None:
            logger.info(f"Loading spatial distances from {spatial_distance_file}")
            self.spatial_distances = self._load_spatial_distances()
            logger.info(f"Loaded {len(self.spatial_distances)} spatial distance pairs")

        super().__init__(str(root), transform=None, pre_transform=None, pre_filter=None)

        # Ensure the processed_dir exists
        if not os.path.exists(self.processed_dir):
            logger.info(f"Creating processed directory at {self.processed_dir}")
            os.makedirs(self.processed_dir)

        if self.force_reprocess == True:
            logger.info("Force reprocessing enabled - cleaning up existing processed files")
            # Delete all previous .pt files
            deleted_count = 0
            for fname in os.listdir(self.processed_dir):
                if fname.startswith("data_") and fname.endswith(".pt"):
                    os.remove(os.path.join(self.processed_dir, fname))
                    deleted_count += 1
            logger.info(f"Deleted {deleted_count} existing processed files")

            if use_selected_features == True:
                logger.info("Starting feature processing...")
                self.process_features()

            elif use_embeddings == True:
                logger.info("Starting embedding processing...")
                self.process_embeddings()
            else:
                logger.info("Starting session processing...")
                self.process_sessions()

    def _load_spatial_distances(self) -> Dict:
        """
        Load spatial distances between electrodes.
        Expected format: csv file that can be loaded into a dictionary or matrix
        representing distances between electrodes.

        Returns:
            Dictionary of distances or adjacency matrix
        """
        logger.info(f"Loading spatial distances from {self.spatial_distance_file}")
        start_time = time.time()
        
        spatial_distances = (
            {}
        )  # Dictionary to store distances between electrodes with keys (tuple) as (ch1, ch2)
        df_distances = pd.read_csv(str(self.spatial_distance_file))
        for _, row in df_distances.iterrows():
            ch1 = row["from"]
            ch2 = row["to"]
            dist = row["distance"]

            spatial_distances[(ch1, ch2)] = dist
            spatial_distances[(ch2, ch1)] = dist

        load_time = time.time() - start_time
        logger.info(f"Loaded {len(spatial_distances)} spatial distances in {load_time:.2f}s")
        return spatial_distances

    def process_embeddings(self):
        """
        Convert precomputed [19, D] embeddings into PyTorch Geometric graph Data objects.
        """
        logger.info("Starting embedding processing...")
        start_time = time.time()

        embeddings = np.load(str(self.embeddings_dir / "embeddings.npy"))
        if not self.is_test:
            labels = np.load(self.embeddings_dir / "labels_embeddings.npy")
            logger.info(f"Loaded embeddings shape: {embeddings.shape}, labels shape: {labels.shape}")
            assert len(embeddings) == len(labels), "Mismatch between embeddings and labels"
        else:
            logger.info(f"Loaded embeddings shape: {embeddings.shape} (test mode - no labels)")

        processed_count = 0
        skipped_count = 0
        
        for i in range(len(embeddings)):
            if i % 100 == 0:
                logger.info(f"Processing embedding {i+1}/{len(embeddings)}")
                
            segment = embeddings[i]  # shape: [19, D]

            # Normalize across nodes (per feature)
            mean = segment.mean(axis=0, keepdims=True)
            std = segment.std(axis=0, keepdims=True) + 1e-6
            segment = (segment - mean) / std

            x = torch.tensor(segment, dtype=torch.float32)  # [19, D]
            edge_index = self._create_edges(segment)

            if edge_index.numel() == 0:
                logger.warning(f"Skipping embedding {i} - no edges created")
                self.ids_to_eliminate.append(i)
                skipped_count += 1
                continue

            # Get the original id from clips DataFrame
            original_id = self.clips.iloc[i]['id']

            # Create Data object with or without labels
            if self.is_test:
                data = Data(x=x, edge_index=edge_index, id=original_id)
            else:
                y = torch.tensor([labels[i]], dtype=torch.float32) # type: ignore
                data = Data(x=x, edge_index=edge_index, y=y, id=original_id)

            torch.save(data, osp.join(self.processed_dir, f"data_{processed_count}.pt"))
            processed_count += 1

        total_time = time.time() - start_time
        logger.info(f"Embedding processing completed in {total_time:.2f}s")
        logger.info(f"Processed {processed_count} embeddings, skipped {skipped_count}")

    def process_features(self):
        """
        Processes extracted features from samples into PyTorch Geometric Data objects.
        """
        logger.info("Starting feature processing...")
        start_time = time.time()

        extracted_features = np.load(self.extracted_features_dir / "X_train_DE.npy")
        logger.info(f"Loaded features shape: {extracted_features.shape}")
        
        processed_count = 0
        skipped_count = 0

        for index, segment_signal in enumerate(extracted_features):
            if index % 100 == 0:
                logger.info(f"Processing feature {index+1}/{len(extracted_features)}")
                
            segment_signal = segment_signal.reshape(self.n_channels, -1)
            # Normalize features
            mean = segment_signal.mean(axis=1, keepdims=True)
            std = segment_signal.std(axis=1, keepdims=True) + 1e-6
            segment_signal = (segment_signal - mean) / std

            x = torch.tensor(segment_signal, dtype=torch.float)
            edge_index = self._create_edges(segment_signal)

            if edge_index.numel() == 0:
                logger.warning(f"Skipping feature {index} - no edges created")
                self.ids_to_eliminate.append(index)
                skipped_count += 1
                continue

            # Get the original id from clips DataFrame
            original_id = self.clips.iloc[index]['id']

            # Create Data object with or without labels
            if self.is_test:
                data = Data(x=x, edge_index=edge_index, id=original_id)
            else:
                y = torch.tensor([self.clips["label"].values[index]], dtype=torch.float)
                data = Data(x=x, edge_index=edge_index, y=y, id=original_id)

            torch.save(data, osp.join(self.processed_dir, f"data_{processed_count}.pt"))
            processed_count += 1

        total_time = time.time() - start_time
        logger.info(f"Feature processing completed in {total_time:.2f}s")
        logger.info(f"Processed {processed_count} features, skipped {skipped_count}")

    def process_sessions(self):
        """
        Processes raw data into PyTorch Geometric Data objects.
        """
        logger.info("Starting session processing...")
        start_time = time.time()

        sessions = list(self.clips.groupby(["patient", "session"]))
        processed_count = 0
        skipped_count = 0

        for session_idx, ((patient, session), session_df) in enumerate(sessions):
            session_start_time = time.time()
            logger.info(f"Processing session {session_idx+1}/{len(sessions)} (Patient {patient}, Session {session})")
            
            session_signal = pd.read_parquet(
                f"{self.signal_folder}/{session_df['signals_path'].values[0]}"
            )

            session_signal = self._preprocess_signal(session_signal)
            logger.info(f"Preprocessed signal shape: {session_signal.shape}")

            for index, row in session_df.iterrows():
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} segments so far")
                    
                start = int(row["start_time"] * self.sampling_rate)
                end = int(row["end_time"] * self.sampling_rate)
                segment_signal = session_signal[start:end].T

                x = torch.tensor(segment_signal, dtype=torch.float)
                edge_index = self._create_edges(segment_signal)

                if edge_index.numel() == 0:
                    logger.warning(f"Skipping segment {index} - no edges created")
                    self.ids_to_eliminate.append(index)
                    skipped_count += 1
                    continue

                # Get the original id from the row
                original_id = row['id']

                # Create Data object with or without labels
                if self.is_test:
                    data = Data(x=x, edge_index=edge_index, id=original_id)
                else:
                    y = torch.tensor([row["label"]], dtype=torch.float)
                    data = Data(x=x, edge_index=edge_index, y=y, id=original_id)

                torch.save(data, osp.join(self.processed_dir, f"data_{processed_count}.pt"))
                processed_count += 1

            session_time = time.time() - session_start_time
            logger.info(f"Session {session_idx+1} processed in {session_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Session processing completed in {total_time:.2f}s")
        logger.info(f"Processed {processed_count} segments, skipped {skipped_count}")

    def _filter_signal(self, signal):
        return time_filtering(
            signal, bp_filter=self.bp_filter, notch_filter=self.notch_filter
        )

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

    def _create_edges(self, signal_data: torch.Tensor) -> torch.Tensor:
        """
        Creates edges between EEG channels based on the specified strategy.

        Args:
            signal_data: Torch tensor containing EEG signals (channels x time)

        Returns:
            torch.Tensor: Edge index tensor of shape [2, num_edges]
        """
        
        if self.top_k == 19: # Fully connected
            logger.debug("Creating fully connected graph")
            edge_list = []
            for i in range(self.n_channels):
                for j in range(self.n_channels):
                    if i != j:
                     edge_list.append([i, j])
            return  torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        elif self.edge_strategy == "spatial":
            logger.debug("Creating edges based on spatial distances")
            return self._create_spatial_edges()
        elif self.edge_strategy == "correlation":
            logger.debug("Creating edges based on correlation")
            return self._create_correlation_edges(signal_data)
        else:
            raise ValueError(f"Unknown edge strategy: {self.edge_strategy}")

    def _create_spatial_edges(self) -> torch.Tensor:
        """
        Creates edges based on spatial distances between electrodes.

        - If top_k is specified: connects each node to its k closest neighbors.
        - Otherwise: connects all pairs with distance <= self.distance_threshold.

        Returns:
            torch.Tensor: edge_index tensor for PyTorch Geometric
        """
        logger.debug("Building spatial distance matrix")
        num_channels = len(self.channels)
        edge_list = []
        adj_dict = defaultdict(list)

        # Build full symmetric distance matrix
        distance_matrix = np.full((num_channels, num_channels), np.inf)
        for i, ch1 in enumerate(self.channels):
            for j, ch2 in enumerate(self.channels):
                if i != j:
                    dist = self.spatial_distances.get(
                        (ch1, ch2), self.spatial_distances.get((ch2, ch1), np.inf)
                    )
                    distance_matrix[i, j] = dist

        if self.top_k:
            logger.debug(f"Creating top-{self.top_k} spatial connections")
            for i in range(num_channels):
                top_indices = np.argsort(distance_matrix[i])[: self.top_k]
                for j in top_indices:
                    if j not in adj_dict[i]:
                        adj_dict[i].append(j)
                    if i not in adj_dict[j]:
                        adj_dict[j].append(i)

            for i, neighbors in adj_dict.items():
                for j in neighbors:
                    edge_list.append([i, j])
        else:
            logger.debug("Creating fully connected spatial graph")
            for i in range(num_channels):
                for j in range(num_channels):
                    if i != j:
                        edge_list.append([i, j])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        logger.debug(f"Created {edge_index.shape[1]} spatial edges")
        return edge_index

    def _create_correlation_edges(self, signal_data: torch.Tensor) -> torch.Tensor:
        """
        Creates edges based on correlation between channel signals.

        - If self.top_k is set: connects each node to its top-k most correlated neighbors (symmetric).
        - Otherwise: connects all pairs with correlation >= self.correlation_threshold (symmetric).

        Args:
            signal_data (torch.Tensor): EEG signal data (channels x time)

        Returns:
            torch.Tensor: Edge index tensor (2, num_edges)
        """
        logger.debug("Computing correlation matrix")
        corr_matrix = torch.abs(torch.corrcoef(signal_data))

        # Eliminate samples with NaN in corr matrix (0 signal)
        if np.isnan(corr_matrix).any():
            logger.warning("Invalid correlations detected (NaN), returning empty edge index")
            return torch.empty((2, 0), dtype=torch.long)
        if np.isinf(corr_matrix).any():
            logger.warning("Invalid correlations detected (Inf), returning empty edge index")
            return torch.empty((2, 0), dtype=torch.long)

        num_channels = len(self.channels)
        edge_list = []

        if self.top_k:
            logger.debug(f"Creating top-{self.top_k} correlation connections")
            adj_dict = defaultdict(set)

            for i in range(num_channels):
                top_indices = np.argsort(-corr_matrix[i])  # descending order
                count = 0
                for j in top_indices:
                    if i != j and count < self.top_k:
                        adj_dict[i].add(j)
                        adj_dict[j].add(i)  # ensure symmetry
                        count += 1

            for i, neighbors in adj_dict.items():
                for j in neighbors:
                    edge_list.append([i, j])

        else:
            logger.debug(f"Creating correlation connections with threshold {self.correlation_threshold}")
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    if corr_matrix[i, j] >= self.correlation_threshold:
                        edge_list.append([i, j])
                        edge_list.append([j, i])

        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        logger.debug(f"Created {edge_index.shape[1]} correlation edges")
        return edge_index

    def len(self) -> int:
        """
        Returns the number of examples in the dataset (number of graphs saved)
        """
        length = len(self.processed_file_names)
        logger.debug(f"Dataset length: {length}")
        return length

    def get(self, idx: int) -> Data:
        """
        Returns the data object at index idx.
        """
        file_path = osp.join(self.processed_dir, f"data_{idx}.pt")
        try:
            data = torch.load(file_path)
            logger.debug(f"Loaded data from {file_path}")
        except Exception as e:
            # Handle PyTorch 2.6+ weights_only error
            if "weights_only" in str(e).lower():
                logger.debug(f"Retrying load with weights_only=False for {file_path}")
                data = torch.load(file_path, weights_only=False)
            else:
                logger.error(f"Error loading {file_path}: {e}")
                raise RuntimeError(f"Error loading processed file {file_path} for index {idx}: {e}")
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
        files = sorted(
            [
                f
                for f in os.listdir(self.processed_dir)
                if f.startswith("data_")
                and f.endswith(".pt")  # Guard against other files
            ]
        )
        logger.debug(f"Found {len(files)} processed files")
        return files
