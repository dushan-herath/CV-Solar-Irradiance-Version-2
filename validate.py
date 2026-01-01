"""
Runs model inference on the validation split and exports:
- denormalized predictions
- denormalized ground-truth targets
- error metrics (MSE, MAE, RMSE) per forecast horizon

All results are saved into a single compressed NumPy file.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IrradianceForecastDataset
from model import ImageEncoder, MultimodalForecaster


@torch.no_grad()
def evaluate(model, loader, device, mean_targets, std_targets):
    """
    Runs forward passes on the validation set and computes
    prediction errors after denormalization.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Iterate over validation batches
    for sky_seq, ts_seq, targets, *_ in tqdm(
        loader, desc="Evaluating", leave=False
    ):
        sky_seq = sky_seq.to(device)
        ts_seq = ts_seq.to(device)
        targets = targets.to(device)

        # Forward pass
        preds = model(sky_seq, ts_seq)

        # Safety check in case horizon lengths do not match exactly
        if preds.shape[1] != targets.shape[1]:
            preds = preds[:, : targets.shape[1], :]

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # Stack all batches into full arrays
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Convert predictions and targets back to physical units
    preds_denorm = preds * std_targets + mean_targets
    targets_denorm = targets * std_targets + mean_targets

    # Compute forecast errors
    errors = preds_denorm - targets_denorm
    mse_per_horizon = np.mean(errors ** 2, axis=0)
    mae_per_horizon = np.mean(np.abs(errors), axis=0)
    rmse_per_horizon = np.sqrt(mse_per_horizon)

    return (
        preds_denorm,
        targets_denorm,
        mse_per_horizon,
        mae_per_horizon,
        rmse_per_horizon,
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()

    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and evaluation settings (should match training)
    CSV_PATH = "dataset_full_1M.csv"
    IMG_SEQ_LEN = 5
    TS_SEQ_LEN = 30
    MAX_HORIZON = 15
    TARGET_DIM = 1
    BATCH_SIZE = 32

    print(f"Exporting predictions and metrics on {DEVICE}")

    # Load normalization statistics saved during training
    if not os.path.exists("norm_stats.json"):
        raise FileNotFoundError("norm_stats.json not found")

    full_norm_stats = json.load(open("norm_stats.json"))
    full_mean = pd.Series(full_norm_stats["mean"])
    full_std = pd.Series(full_norm_stats["std"])
    normalization_stats = {"mean": full_mean, "std": full_std}

    # Target variable configuration
    TARGET_NAMES = ["ghi"]
    mean_targets = np.array(
        [full_mean[n] for n in TARGET_NAMES]
    ).reshape(1, 1, TARGET_DIM)
    std_targets = np.array(
        [full_std[n] for n in TARGET_NAMES]
    ).reshape(1, 1, TARGET_DIM)

    # Initialize validation dataset and loader
    val_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=MAX_HORIZON,
        normalization_stats=normalization_stats,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    print(
        f"Validation dataset loaded: {len(val_ds)} samples, "
        f"horizon={MAX_HORIZON}"
    )

    # Build model using the same architecture as training
    sky_encoder = ImageEncoder(
        model_name="resnet18",
        pretrained=True,
        freeze=True,
    )

    model = MultimodalForecaster(
        sky_encoder=sky_encoder,
        ts_feat_dim=len(full_mean),
        ts_embed_dim=64,
        fused_dim=256,
        horizon=MAX_HORIZON,
        target_dim=TARGET_DIM,
    ).to(DEVICE)

    # Load the best checkpoint from training
    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found")

    model.load_state_dict(
        torch.load("best_model.pth", map_location=DEVICE)
    )
    print("Loaded trained model weights")

    # Run evaluation
    preds_denorm, targets_denorm, mse, mae, rmse = evaluate(
        model, val_loader, DEVICE, mean_targets, std_targets
    )

    # Save all outputs into a single compressed file
    save_path = "forecast_results.npz"
    np.savez_compressed(
        save_path,
        preds=preds_denorm,
        targets=targets_denorm,
        mse=mse,
        mae=mae,
        rmse=rmse,
        target_names=np.array(TARGET_NAMES),
    )

    print(f"\nResults saved to {save_path}")
    print(
        f"Prediction shape: {preds_denorm.shape}, "
        f"Target shape: {targets_denorm.shape}, "
        f"Metric shape: {mse.shape}"
    )
    print("Evaluation complete")
