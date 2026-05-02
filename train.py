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
from tqdm import tqdm

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

RUN_NAME = "main_run"
SAVE_DIR = os.path.join(BASE_SAVE_DIR, RUN_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Run directory connected: {SAVE_DIR}")

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
        "best_loss": getattr(model, "best_loss", loss), # Optional to carry forward
        "config": config,
        "environment": get_environment_info()
    }

    # 🔹 Atomic write for local save
    tmp_local = local_path + ".tmp"
    torch.save(checkpoint, tmp_local)
    os.replace(tmp_local, local_path)
    
    # 🔹 Atomic write for Drive save (prevents corruption)
    tmp_final = final_path + ".tmp"
    shutil.copy(local_path, tmp_final)
    os.replace(tmp_final, final_path)
    print(f"✅ Full checkpoint saved → {final_path}")

def main():
    print("Starting Training Execution (Level-3 Safety)...")
    
    SESSION_START = time.time()
    MAX_RUNTIME = 60 * 60  # 60 minutes safe cutoff
    
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
    
    # In Colab, we can push batch sizes and workers up for high speed
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True if 2 > 0 else False,
        prefetch_factor=2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    model = SimpleCNN(num_classes=len(species_list)).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # 🔹 Auto-Resume Checkpoint Logic
    start_epoch = 0
    best_loss = float("inf")
    
    # 🔹 Use checkpoint.pth consistently everywhere
    checkpoint_path = os.path.join(SAVE_DIR, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("\n🔁 Resuming from checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Checkpoint load failed: {e}")
            print("Starting fresh training...")
            start_epoch = 0
            best_loss = float("inf")
        
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Start epoch: {start_epoch}")
    print(f"Best loss: {best_loss}")
    
    if start_epoch >= config["num_epochs"]:
        print("Training already completed.")
        return
        
    print("\nStarting Training...")
    
    # 🔹 4. Training Loop (with auto-save + crash safety)
    # Initialize safe defaults to prevent UnboundLocalError
    epoch = start_epoch - 1 # Fallback if loop doesn't run
    avg_loss = float("inf")

    try:
        for epoch in range(start_epoch, config["num_epochs"]):
            print(f"Epoch {epoch} → {len(train_loader)} batches")
            model.train()
            batches_processed = 0
            total_loss = 0.0
            
            t0 = time.time()
            try:
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']-1}")):
                    batches_processed += 1
                    if (time.time() - SESSION_START) > MAX_RUNTIME:
                        print("\n⏳ Time limit approaching — exiting safely...")
                        break
    
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    if epoch == start_epoch and batch_idx == 0:
                        print(f"\nDevice check → {inputs.device}")
                        print(f"Device: {inputs.device}, Batch shape: {inputs.shape}")
                        print(f"First batch load time: {time.time() - t0:.2f}s")
                    
                    if batch_idx % 200 == 0:
                        print(f"Batch {batch_idx} running on {inputs.device}")
                    
                    optimizer.zero_grad()
                    
                    try:
                        outputs = model(inputs)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print("CUDA OOM — reducing batch size required")
                            raise e
                        else:
                            raise e
                            
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
            except RuntimeError as e:
                print(f"DataLoader failure: {e}")
                raise e
                
            if batches_processed > 0:
                avg_loss = total_loss / batches_processed
            else:
                print("⚠️ No batches processed — skipping loss computation")
                continue
                
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

            # 🔹 Save BEST model
            if avg_loss < best_loss:
                best_loss = avg_loss
                model.best_loss = best_loss # Attach so checkpointing logic natively grabs it
                save_full_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    config,
                    name="best_model.pth"
                )
                print("Best model updated")

            # 🔹 Save constant running checkpoint tracking current state
            save_full_checkpoint(
                model,
                optimizer,
                epoch,
                avg_loss,
                config,
                name="checkpoint.pth"
            )

    finally:
        print("\n⚠️ Training ended or interrupted → Saving final state...")
        try:
            if epoch >= 0:
                save_full_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_loss,
                    config,
                    name="checkpoint.pth"
                )
                checkpoint_path = os.path.join(SAVE_DIR, "checkpoint.pth")
                if os.path.exists(checkpoint_path):
                    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                    print(f"Checkpoint size: {size_mb:.2f} MB")
            print("Training Loop Terminated Successfully.")
        except Exception as e:
            print(f"❌ Final save failed: {e}")

if __name__ == "__main__":
    main()
