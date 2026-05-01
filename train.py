import os
import sys
import ast
import time
import shutil
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

# Import our custom files
from dataset import BirdCLEFDataset
from model import SimpleCNN

# 🔹 1. Clean Path Configurations
# Colab uses /content, local uses current directory
DATA_DIR = "/content" if os.path.exists("/content") else "."
AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")

# Support both naming conventions for Kaggle CSVs
if os.path.exists(os.path.join(DATA_DIR, "train_metadata.csv")):
    CSV_PATH = os.path.join(DATA_DIR, "train_metadata.csv")
else:
    CSV_PATH = os.path.join(DATA_DIR, "train.csv")

# 🔹 2. Setup Save Directory (Drive ONLY when in Colab)
if os.path.exists("/content/drive/MyDrive"):
    BASE_SAVE_DIR = "/content/drive/MyDrive/birdclef_models"
    LOCAL_TMP_DIR = "/content"
    print("Saving models to Google Drive")
else:
    # Use /content if in Colab, otherwise local
    BASE_SAVE_DIR = "/content/birdclef_models" if os.path.exists("/content") else "./birdclef_models"
    LOCAL_TMP_DIR = "/content" if os.path.exists("/content") else "."
    print("⚠️ Drive not mounted — saving locally")

RUN_NAME = f"run_{int(time.time())}"
SAVE_DIR = os.path.join(BASE_SAVE_DIR, RUN_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Run directory created: {SAVE_DIR}")

# 🔹 3. Add FULL config + environment capture
config = {
    "learning_rate": 0.001,
    "batch_size": 8,
    "num_epochs": 10,
    "sample_rate": 32000,
    "model_name": "CNN_Baseline_v1"
}

def get_environment_info():
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform()
    }

# 🔹 3. FULL Level-3 Save Function
def save_full_checkpoint(model, optimizer, epoch, loss, config, name="final_checkpoint.pth"):
    # Save to fast local disk first
    local_path = os.path.join(LOCAL_TMP_DIR, name)
    final_path = os.path.join(SAVE_DIR, name)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "config": config,
        "environment": get_environment_info()
    }

    torch.save(checkpoint, local_path)
    
    # Then copy to Drive (this avoids Google Drive bottleneck during training)
    shutil.copy(local_path, final_path)
    print(f"✅ Full checkpoint saved → {final_path}")

def main():
    print("Starting Training Execution (Level-3 Safety)...")
    
    print("Audio dir:", AUDIO_DIR)
    print("CSV path:", CSV_PATH)
    
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please ensure your dataset is extracted here.")
        return

    df = pd.read_csv(CSV_PATH)

    primary = set(df["primary_label"].unique())

    secondary = set()
    for labels in df["secondary_labels"]:
        if pd.isna(labels):
            continue
        try:
            parsed = ast.literal_eval(labels)
            secondary.update(parsed)
        except:
            continue

    species_list = sorted(list(primary.union(secondary)))
    print("Number of species:", len(species_list))
    
    dataset = BirdCLEFDataset(
        metadata_path=CSV_PATH,
        audio_dir=AUDIO_DIR,
        species_list=species_list
    )
    
    # In Colab + torch + librosa, we batch load to prevent GIL lock
    train_loader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model = SimpleCNN(num_classes=len(species_list)).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print("\nStarting Training...")
    
    # 🔹 4. Training Loop (with auto-save + crash safety)
    # Initialize safe defaults to prevent UnboundLocalError
    epoch = -1
    best_loss = float("inf")
    avg_loss = float("inf")

    try:
        for epoch in range(config["num_epochs"]):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

            # 🔹 Save BEST model
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_full_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    config,
                    name="best_model.pth"
                )

            # 🔹 Save EACH epoch (optional but powerful)
            save_full_checkpoint(
                model,
                optimizer,
                epoch,
                avg_loss,
                config,
                name=f"checkpoint_epoch_{epoch}.pth"
            )

    finally:
        print("\n⚠️ Training ended or interrupted → Saving final state...")
        try:
            save_full_checkpoint(
                model,
                optimizer,
                epoch,
                avg_loss,
                config,
                name="final_model.pth"
            )
            print("Training Loop Terminated Successfully.")
        except Exception as e:
            print(f"❌ Final save failed: {e}")

if __name__ == "__main__":
    main()
