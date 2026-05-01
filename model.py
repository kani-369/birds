import torch
import torch.nn as nn

# BirdCLEF CNN baseline model

# Task:
# Build a simple CNN for spectrogram classification

# Input:
# (batch_size, 1, 128, time_steps)

# Architecture:
# - Conv2D -> ReLU -> MaxPool
# - Conv2D -> ReLU -> MaxPool
# - Flatten -> Fully connected layer

# Output:
# - Multi-label classification
# - Output size = number of species
# - No softmax

# Framework:
# PyTorch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ✨ PRO TIP: AdaptiveMaxPool2d handles variable time_steps automatically.
        # It forces the output map to a fixed (2, 2) shape before flattening, 
        # so you NEVER get shape mismatch errors in the Linear layer!
        self.adaptive_pool = nn.AdaptiveMaxPool2d((2, 2))
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Fully connected layer
        # Input mapped features: 32 channels * 2 * 2 mapped adaptive pool output = 128
        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # Forward pass through architecture
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        # Returns raw logits (no softmax/sigmoid here)
        return x
