import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as TF


class IrradianceForecastDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        val_ratio: float = 0.2,
        img_seq_len: int = 5,
        ts_seq_len: int = 30,
        horizon: int = 25,
        feature_cols=None,
        target_cols=None,
        img_size: int = 64,
        time_col: str = "timestamp",
        normalization_stats: dict = None,
    ):
        # -------------------------------
        # Load full dataset
        # -------------------------------
        df = pd.read_csv(csv_path)
        n = len(df)

        # -------------------------------
        # Train / validation split
        # -------------------------------
        split_idx = int(n * (1 - val_ratio))
        if split == "train":
            self.df = df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be either 'train' or 'val'")

        # -------------------------------
        # Configuration
        # -------------------------------
        self.split = split
        self.img_seq_len = img_seq_len
        self.ts_seq_len = ts_seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.time_col = time_col

        self.feature_cols = feature_cols or ["ghi", "dni", "dhi"]
        self.target_cols = target_cols or ["ghi"]

        # Image path columns
        self.sky_col = "raw_image_path"
        self.flow_col = "optical_flow_image_path"

        self.max_lookback = max(img_seq_len, ts_seq_len)

        # -------------------------------
        # Timestamp handling
        # -------------------------------
        if self.time_col in self.df.columns:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        # -------------------------------
        # Feature normalization
        # -------------------------------
        if split == "train":
            mean = self.df[self.feature_cols].mean()
            std = self.df[self.feature_cols].std()
            self.normalization_stats = {"mean": mean, "std": std}
        else:
            if normalization_stats is None:
                raise ValueError("Validation split requires normalization statistics")
            self.normalization_stats = normalization_stats
            mean = normalization_stats["mean"]
            std = normalization_stats["std"]

        self.df[self.feature_cols] = (self.df[self.feature_cols] - mean) / std

        # -------------------------------
        # Image preprocessing
        # -------------------------------
        self.img_resize = transforms.Resize((img_size, img_size))
        self.flow_resize = transforms.Resize((img_size, img_size))

        self.img_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # -------------------------------
        # Logging
        # -------------------------------
        print(f"\nDataset initialized ({split.upper()}): {len(self)} samples")
        print(
            f"Image sequence length: {img_seq_len}, "
            f"Time-series length: {ts_seq_len}, "
            f"Forecast horizon: {horizon}"
        )

    def __len__(self):
        return len(self.df) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        # -------------------------------
        # Select windows
        # -------------------------------
        img_window = self.df.iloc[
            idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len
        ]
        ts_window = self.df.iloc[idx : idx + self.ts_seq_len]
        target_window = self.df.iloc[
            idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon
        ]

        # -------------------------------
        # One random rotation per sequence
        # -------------------------------
        if self.split == "train":
            angle = random.uniform(-180, 180)
        else:
            angle = 0.0

        # -------------------------------
        # Load sky + optical flow images
        # -------------------------------
        img_seq = []

        for sky_p, flow_p in zip(
            img_window[self.sky_col].values,
            img_window[self.flow_col].values,
        ):
            # ---- Sky image ----
            sky_img = Image.open(sky_p).convert("RGB")
            sky_img = self.img_resize(sky_img)

            # ---- Optical flow image ----
            flow_img = Image.open(flow_p).convert("RGB")
            flow_img = self.flow_resize(flow_img)

            # ---- Shared augmentation ----
            if angle != 0.0:
                sky_img = TF.rotate(
                    sky_img,
                    angle=angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )
                flow_img = TF.rotate(
                    flow_img,
                    angle=angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )

            # ---- To tensor ----
            sky_img = TF.to_tensor(sky_img)
            sky_img = self.img_normalize(sky_img)

            flow_img = TF.to_tensor(flow_img)
            # NOTE: no normalization for optical flow

            # ---- Concatenate (6 channels) ----
            img_6ch = torch.cat([sky_img, flow_img], dim=0)
            img_seq.append(img_6ch)

        sky_seq = torch.stack(img_seq)  # (T, 6, H, W)

        # -------------------------------
        # Time-series inputs
        # -------------------------------
        ts_seq = torch.tensor(
            ts_window[self.feature_cols].values,
            dtype=torch.float32,
        )

        # -------------------------------
        # Targets
        # -------------------------------
        target_seq = torch.tensor(
            target_window[self.target_cols].values,
            dtype=torch.float32,
        )

        # -------------------------------
        # Timestamps
        # -------------------------------
        ts_time = (
            ts_window[self.time_col]
            .dt.floor("s")
            .astype(str)
            .tolist()
        )
        tgt_time = (
            target_window[self.time_col]
            .dt.floor("s")
            .astype(str)
            .tolist()
        )

        return sky_seq, ts_seq, target_seq, ts_time, tgt_time


# ======================================================================
# Debug / Visualization
# ======================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    CSV_PATH = "dataset_full_1M.csv"

    dataset = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        img_seq_len=5,
        ts_seq_len=30,
        horizon=30,
    )

    sky_seq, ts_seq, target_seq, ts_time, tgt_time = dataset[2200]

    print("Sky sequence shape:", sky_seq.shape)  # (T, 6, H, W)
    print("TS shape:", ts_seq.shape)
    print("Target shape:", target_seq.shape)

    # Visualize RGB only
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    T = sky_seq.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))

    for i in range(T):
        rgb = sky_seq[i, :3]
        rgb_denorm = (rgb * std + mean).clamp(0, 1)
        axes[0, i].imshow(rgb_denorm.permute(1, 2, 0))
        axes[0, i].set_title("RGB")
        axes[0, i].axis("off")

        flow = sky_seq[i, 3:]
        axes[1, i].imshow(flow.permute(1, 2, 0))
        axes[1, i].set_title("Optical Flow")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()
