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
import numpy as np
import torch
import torchaudio

def process_audio(file_path, sr=32000, segment_duration=5.0, n_mels=128, n_fft=2048, hop_length=512, return_single_random=False):
    """
    Reads an audio file, splits it into fixed duration segments, 
    and converts them to normalized mel spectrograms.
    """
    if return_single_random:
        # SUPER-FAST PATH: Bypass librosa/scipy entirely to prevent hanging CPU threads!
        # torchaudio compiles natively in C++ and executes FFT hundreds of times faster.
        
        # 1. Load just 5 seconds instantly using torchaudio
        num_frames = int(sr * segment_duration)
        try:
            # Try to load exactly what we need
            waveform, sample_rate = torchaudio.load(file_path, num_frames=num_frames)
        except Exception:
            # Fallback
            y, sample_rate = librosa.load(file_path, sr=sr, offset=0.0, duration=segment_duration)
            waveform = torch.from_numpy(y).unsqueeze(0)

        # Resample if necessary
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
            waveform = resampler(waveform)

        # Convert to Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad
        expected_samples = int(sr * segment_duration)
        if waveform.shape[1] < expected_samples:
            padding = expected_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Fast GPU/C++ natively optimized Mel Spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        mel_spec = mel_transform(waveform)
        
        # Power to DB and Normalize
        amp_to_db = torchaudio.transforms.AmplitudeToDB()
        mel_spec_db = amp_to_db(mel_spec)
        
        mel_min = mel_spec_db.min()
        mel_max = mel_spec_db.max()
        eps = 1e-8
        
        normalized_mel_spec = (mel_spec_db - mel_min) / (mel_max - mel_min + eps)
        
        return normalized_mel_spec.numpy()  # Returns shape (1, 128, X)

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

