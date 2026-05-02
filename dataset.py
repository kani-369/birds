import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from data_pipeline import process_audio

# BirdCLEF Dataset Class

# Task:
# Build a PyTorch Dataset class for BirdCLEF

# Requirements:
# - Use process_audio() from data_pipeline.py
# - Load metadata CSV (file_path + labels)
# - For each audio file:
#     - Convert into 5-sec spectrogram segments
#     - Assign labels (multi-label)
# - Return (spectrogram, label_vector)


class BirdCLEFDataset(Dataset):
    def __init__(self, metadata_path, audio_dir, species_list):
        """
        Args:
            metadata_path (str): Path to the metadata CSV.
            audio_dir (str): Directory with audio files.
            species_list (list): List of all unique species for multi-hot encoding.
        """
        self.df = pd.read_csv(metadata_path)
        self.audio_dir = audio_dir
        self.num_classes = len(species_list)
        
        # Create a fast O(1) lookup mapping from species name to integer index
        self.species_to_idx = {species: idx for idx, species in enumerate(species_list)}

    def __len__(self):
        return len(self.df)

    def _get_multi_hot_labels(self, primary_label, secondary_labels=None):
        """Creates a multi-hot encoded vector for the given labels."""
        label_vector = np.zeros(self.num_classes, dtype=np.float32)
        
        # Set primary label (Always 1.0)
        if primary_label in self.species_to_idx:
            label_vector[self.species_to_idx[primary_label]] = 1.0
            
        # Set secondary labels if they exist (Also 1.0 for multi-label)
        if pd.notna(secondary_labels):
            # Standard BirdCLEF Kaggle format for secondary labels is usually a string like "['sp1', 'sp2']"
            if isinstance(secondary_labels, str):
                cleaned = secondary_labels.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                if cleaned.strip():
                    sec_list = [s.strip() for s in cleaned.split(',')]
                    for sec_label in sec_list:
                        if sec_label in self.species_to_idx:
                            label_vector[self.species_to_idx[sec_label]] = 1.0
                            
        return label_vector

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # In typical BirdCLEF datasets, the path is in 'filename' or 'audio_id'
        # We'll use a generic approach based on standard naming 'filename'
        audio_path = os.path.join(self.audio_dir, row.get('filename', ''))
        
        # 1. Convert into a single 5-sec spectrogram segment (Extremely fast via TorchAudio)
        spectrograms = process_audio(audio_path, return_single_random=True)
        
        # 🔥 Torchaudio array is already (1, 128, time_steps)
        segment = spectrograms[0] if len(spectrograms.shape) > 3 else spectrograms
        
        # 2. Assign labels (multi-label multi-hot encoding)
        primary = row.get('primary_label', '')
        secondary = row.get('secondary_labels', '')
        
        label_vector = self._get_multi_hot_labels(primary, secondary)
        
        # Convert to PyTorch tensors (and avoid CPU bottleneck with astype)
        segment = np.array(segment).astype("float32")
        label_vector = np.array(label_vector).astype("float32")
        
        segment_tensor = torch.tensor(segment, dtype=torch.float32)
        label_tensor = torch.tensor(label_vector, dtype=torch.float32)
        
        return segment_tensor, label_tensor
