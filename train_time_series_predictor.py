import os
import json
import torch
import contextlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from dataset import IrradianceForecastDataset
from model_time_series_predictor import TimeSeriesForecaster


# ==========================================================
# Dataset Wrapper (TIME-SERIES ONLY)
# ==========================================================
class TSForecastWrapper(torch.utils.data.Dataset):
    """
    Returns:
        past_ts   : (T, D)
        future_ts : (H, 1)
    """
    def __init__(self, base_ds: IrradianceForecastDataset):
        self.ds = base_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Dataset structure (based on your sky script)
        sky_seq, ts_past, ghi_future, _, _ = self.ds[idx]

        return ts_past.float(), ghi_future.float()


# ==========================================================
# Training / Validation
# ==========================================================
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=True)

    for i, (past_ts, future_ts) in enumerate(loop):
        past_ts = past_ts.to(device)
        future_ts = future_ts.to(device)

        optimizer.zero_grad()

        ac = (
            torch.cuda.amp.autocast()
            if device.type == "cuda"
            else contextlib.nullcontext()
        )

        with ac:
            preds = model(past_ts)
            loss = criterion(preds, future_ts)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(
            {
                "avg_loss": total_loss / (i + 1),
                "batch_loss": loss.item(),
            }
        )

    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    loop = tqdm(loader, desc="Validation", leave=True)

    with torch.no_grad():
        for i, (past_ts, future_ts) in enumerate(loop):
            past_ts = past_ts.to(device)
            future_ts = future_ts.to(device)

            preds = model(past_ts)
            loss = criterion(preds, future_ts)

            total_loss += loss.item()
            loop.set_postfix(
                {
                    "avg_loss": total_loss / (i + 1),
                    "batch_loss": loss.item(),
                }
            )

    return total_loss / len(loader)


# ==========================================================
# Utilities
# ==========================================================
def plot_losses(train_losses, val_losses, save_path="training_curve_ts.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss (MAE)")
    plt.title("Time-Series Forecasting")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot → {save_path}")


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved → {filename}")


def load_checkpoint(filename, model, optimizer, device):
    ckpt = torch.load(filename, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return (
        ckpt["epoch"],
        ckpt["best_val_loss"],
        ckpt["train_losses"],
        ckpt["val_losses"],
    )


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    CSV_PATH = "dataset_full_1M.csv"

    BATCH_SIZE = 64
    NUM_EPOCHS = 40

    TS_SEQ_LEN = 30
    HORIZON = 25
    TS_FEAT_DIM = 3   # ghi, dni, dhi, temp, pressure, etc.

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    # ------------------------------------------------------
    # Datasets
    # ------------------------------------------------------
    train_base = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
    )

    val_base = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
        normalization_stats=train_base.normalization_stats,
    )

    # Save normalization stats
    json.dump(
        {
            "mean": train_base.normalization_stats["mean"].to_dict(),
            "std": train_base.normalization_stats["std"].to_dict(),
        },
        open("norm_stats_ts.json", "w"),
        indent=4,
    )

    train_ds = TSForecastWrapper(train_base)
    val_ds = TSForecastWrapper(val_base)

    # 🔎 SANITY CHECK (CRITICAL)
    x, y = train_ds[0]
    print("Sanity check:")
    print("Past TS shape:", x.shape)     # (T, 11)
    print("Future TS shape:", y.shape)   # (H, 1)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    model = TimeSeriesForecaster(
        ts_feat_dim=TS_FEAT_DIM,
        ts_embed_dim=64,
        model_dim=128,
        horizon=HORIZON,
        target_dim=1,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    # ------------------------------------------------------
    # Optimizer / Loss
    # ------------------------------------------------------
    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # ------------------------------------------------------
    # Resume
    # ------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    if os.path.exists("checkpoint_ts.pth"):
        (
            start_epoch,
            best_val_loss,
            train_losses,
            val_losses,
        ) = load_checkpoint(
            "checkpoint_ts.pth", model, optimizer, DEVICE
        )
        print(f"Resumed from epoch {start_epoch + 1}")

    # ------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            DEVICE,
            scaler,
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_ts_model.pth")
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
            "checkpoint_ts.pth",
        )

    plot_losses(train_losses, val_losses)

    print("\nTraining complete")
    print(f"Best Validation Loss: {best_val_loss:.5f}")
