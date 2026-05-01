import os
import numpy as np
from scipy.io import wavfile
from data_pipeline import process_audio

# Test Script for BirdCLEF Data Pipeline

file_path = "dummy_audio.wav"

# Create a dummy 12-second audio file if it doesn't already exist
if not os.path.exists(file_path):
    print(f"Creating a 12-second dummy audio file: {file_path}")
    sr = 32000
    # Generate 12 seconds of random noise (standard sample rate)
    dummy_audio = np.random.uniform(-1, 1, sr * 12).astype(np.float32)
    wavfile.write(file_path, sr, dummy_audio)

print("Running process_audio pipeline...")
specs = process_audio(file_path)

print("Number of segments:", len(specs))
print("Shape of one segment:", specs[0].shape)

# Simulating what the dataset does for a single segment output
import random
segment = random.choice(specs)
segment = segment[np.newaxis, :, :]
print("Shape of dataset __getitem__ segment output:", segment.shape)
print("Completed successfully! ✅")
