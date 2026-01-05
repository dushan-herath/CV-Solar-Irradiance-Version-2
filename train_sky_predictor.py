import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import contextlib

from dataset import IrradianceForecastDataset
from model_sky_predictor import SkyFutureImagePredictor


# ----------------------------------------------------------
# Dataset wrapper to load FUTURE sky frames as targets
# ----------------------------------------------------------
class FutureImageWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds: IrradianceForecastDataset):
        self.ds = base_ds
        self.df = base_ds.df
        self.sky_col = base_ds.sky_col
        self.img_transform = base_ds.img_transform
        self.horizon = base_ds.horizon
        self.ts_seq_len = base_ds.ts_seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sky_seq, _, _, _, _ = self.ds[idx]

        start = idx + self.ts_seq_len
        end = start + self.horizon

        future_df = self.df.iloc[start:end]

        future_imgs = torch.stack([
            self.img_transform(Image.open(p).convert("RGB"))
            for p in future_df[self.sky_col].values
        ])

        return sky_seq, future_imgs


# ----------------------------------------------------------
# Training / Validation loops
# ----------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, horizon):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, total=len(loader), desc="Training", leave=True)

    for i, (past_imgs, future_imgs) in enumerate(loop):
        past_imgs = past_imgs.to(device)
        future_imgs = future_imgs.to(device)

        optimizer.zero_grad()

        # -------- autocast fallback (CUDA only) --------
        if device.type == "cuda":
            ac = torch.cuda.amp.autocast()
        else:
            ac = contextlib.nullcontext()

        with ac:
            preds = model(
                past_imgs,
                future_imgs=future_imgs,   # teacher forcing
                horizon=horizon,
            )
            loss = criterion(preds, future_imgs)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (i + 1)

        loop.set_postfix(
            {"avg_loss": avg_loss, "batch_loss": loss.item()},
            refresh=True
        )

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device, horizon):
    model.eval()
    total_loss = 0.0

    loop = tqdm(loader, total=len(loader), desc="Validation", leave=True)

    with torch.no_grad():
        for i, (past_imgs, future_imgs) in enumerate(loop):
            past_imgs = past_imgs.to(device)
            future_imgs = future_imgs.to(device)

            preds = model(
                past_imgs,
                future_imgs=None,   # autoregressive rollout
                horizon=horizon,
            )

            loss = criterion(preds, future_imgs)

            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)

            loop.set_postfix(
                {"avg_loss": avg_loss, "batch_loss": loss.item()},
                refresh=True
            )

    return total_loss / len(loader)


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------
def plot_losses(train_losses, val_losses, save_path="training_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to {save_path}")


def save_checkpoint(state, filename="checkpoint_sky.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {state['epoch'] + 1} -> {filename}")


def load_checkpoint(filename, model, optimizer, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]

    print(
        f"Resumed from checkpoint: epoch {epoch + 1} | "
        f"best val loss = {best_val_loss:.5f}"
    )
    return epoch, best_val_loss, train_losses, val_losses


# ----------------------------------------------------------
# Main entry point
# ----------------------------------------------------------
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()

    CSV_PATH = "dataset_full_1M.csv"
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    IMG_SEQ_LEN = 5
    TS_SEQ_LEN = 30
    HORIZON = 5
    IMG_SIZE = 64

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    # -----------------------------
    # Load datasets
    # -----------------------------
    train_base = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
        img_size=IMG_SIZE,
    )

    val_base = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
        img_size=IMG_SIZE,
        normalization_stats=train_base.normalization_stats,
    )

    # Save normalization stats
    json.dump(
        {
            "mean": train_base.normalization_stats["mean"].to_dict(),
            "std": train_base.normalization_stats["std"].to_dict(),
        },
        open("norm_stats.json", "w"),
        indent=4,
    )

    train_ds = FutureImageWrapper(train_base)
    val_ds = FutureImageWrapper(val_base)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # -----------------------------
    # Build model
    # -----------------------------
    model = SkyFutureImagePredictor(
        img_channels=3,
        hidden_dims=[64, 128, 256],
        lstm_hidden=[256, 256],
        teacher_forcing=True,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"Model ready | Total: {total_params/1e6:.2f}M | "
        f"Trainable: {trainable_params/1e6:.2f}M"
    )

    # -----------------------------
    # Loss & optimizer
    # -----------------------------
    criterion = nn.L1Loss()

    # ---- robust ConvLSTM detection ----
    if hasattr(model, "convlstm"):
        convlstm_params = list(model.convlstm.parameters())
    elif hasattr(model, "convlstm_stack"):
        convlstm_params = list(model.convlstm_stack.parameters())
    else:
        print("WARNING: ConvLSTM module not found — using single LR group")
        convlstm_params = []

    # preferred: grouped LR
    if len(convlstm_params) > 0:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": 3e-4},
                {"params": convlstm_params, "lr": 5e-4},
                {"params": model.decoder.parameters(), "lr": 5e-4},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=4e-4,
            weight_decay=1e-4,
        )

    # ---- GradScaler (CUDA only) ----
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # -----------------------------
    # Resume checkpoint if exists
    # -----------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    if os.path.exists("checkpoint_sky.pth"):
        start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint(
            "checkpoint_sky.pth", model, optimizer, DEVICE
        )
    else:
        print("Starting new training session")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, scaler, HORIZON
        )

        val_loss = validate_one_epoch(
            model, val_loader, criterion, DEVICE, HORIZON
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_sky_model.pth")
            print("Best model updated")

        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            filename="checkpoint_sky.pth",
        )

    plot_losses(train_losses, val_losses)
    print("\nTraining complete")
    print(f"Best Validation Loss: {best_val_loss:.5f}")
