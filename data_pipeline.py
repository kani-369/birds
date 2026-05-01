# BirdCLEF 2026 - Data Pipeline

# Task:
# Build a data preprocessing pipeline for BirdCLEF competition.

# Problem:
# - Input: Continuous audio recordings
# - Each prediction is for a 5-second window
# - Multi-label classification (multiple species can be present)

# Requirements:
# 1. Load audio files from /kaggle/input/birdclef-2026/
# 2. Split audio into 5-second segments
# 3. Convert each segment into mel spectrogram
# 4. Normalize spectrogram
# 5. Handle variable length audio (pad or trim)
# 6. Return (spectrogram, label vector)

# Labels:
# - Read from metadata CSV
# - Multi-hot encoding for species

# Libraries:
# - librosa
# - numpy
# - torch (optional)

# Goal:
# Prepare clean input features for model training

import librosa
import subprocess
import numpy as np

def get_file_duration(file_path):
    """Uses raw ffprobe for instant header reads without opening the file."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        return float(subprocess.check_output(cmd, stderr=subprocess.STDOUT).strip())
    except Exception:
        return 5.0 # Safe fallback

def process_audio(file_path, sr=32000, segment_duration=5.0, n_mels=128, n_fft=2048, hop_length=512, return_single_random=False):
    """
    Reads an audio file, splits it into fixed duration segments, 
    and converts them to normalized mel spectrograms.
    """
    # If training, only load a single random 5-second slice instead of decoding 5 minutes of audio!
    if return_single_random:
        try:
            total_duration = get_file_duration(file_path)
            max_offset = max(0, total_duration - segment_duration)
            offset = np.random.uniform(0, max_offset)
            y, sr = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)
        except Exception:
            y, sr = librosa.load(file_path, sr=sr, duration=segment_duration)
            
        chunk_samples = int(sr * segment_duration)
        if len(y) < chunk_samples:
            y = np.pad(y, (0, chunk_samples - len(y)), mode='constant')
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        eps = 1e-8
        normalized_mel_spec = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + eps)
        
        return np.array([normalized_mel_spec])

    # Step 1: Load audio file
    # sr=32000 is a common sample rate for BirdCLEF
    y, sr = librosa.load(file_path, sr=sr)
    
    chunk_samples = int(sr * segment_duration)
    total_samples = len(y)
    
    spectrograms = []
    
    # Step 2: Split audio into 5-second chunks
    for start in range(0, total_samples, chunk_samples):
        chunk = y[start:start + chunk_samples]
        
        # Handle variable length audio (pad if the last chunk is shorter than 5 seconds)
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
        # Step 3: Convert each chunk into mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=chunk, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Convert power to decibels (log scale)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Step 4: Normalize output
        # Min-max normalization for the current segment to range [0, 1]
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        
        # Add epsilon to prevent division by zero (improves stability)
        eps = 1e-8
        normalized_mel_spec = (mel_spec_db - min_val) / (max_val - min_val + eps)
            
        spectrograms.append(normalized_mel_spec)
        
    return np.array(spectrograms)

