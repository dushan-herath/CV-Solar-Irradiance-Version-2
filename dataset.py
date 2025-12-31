import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


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
        img_size: int = 224,
        time_col: str = "timestamp",
        normalization_stats: dict = None,
    ):
        # Load full dataset from CSV
        df = pd.read_csv(csv_path)
        n = len(df)

        # Split dataset into training and validation sets
        split_idx = int(n * (1 - val_ratio))

        if split == "train":
            self.df = df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be either 'train' or 'val'")

        # Store configuration parameters
        self.split = split
        self.img_seq_len = img_seq_len
        self.ts_seq_len = ts_seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.time_col = time_col

        # Define input features and prediction targets
        self.feature_cols = feature_cols or ["ghi", "dni", "dhi"]
        self.target_cols = target_cols or ["ghi"]

        # Column name containing sky image file paths
        self.sky_col = "raw_image_path"

        # Maximum lookback needed to safely index sequences
        self.max_lookback = max(img_seq_len, ts_seq_len)

        # Convert timestamp column to datetime format
        if self.time_col in self.df.columns:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        # Normalize input features using training statistics
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

        # Image preprocessing pipeline for sky images
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Log dataset configuration
        print(f"\nDataset initialized ({split.upper()}): {len(self)} samples")
        print(f"Image sequence length: {img_seq_len}, "
              f"Time-series length: {ts_seq_len}, "
              f"Forecast horizon: {horizon}")

    def __len__(self):
        # Total number of valid samples available
        return len(self.df) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        # Select image window aligned to the end of the time-series window
        img_window = self.df.iloc[
            idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len
        ]

        # Select historical time-series window
        ts_window = self.df.iloc[idx : idx + self.ts_seq_len]

        # Select future target window for forecasting
        target_window = self.df.iloc[
            idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon
        ]

        # Load and preprocess sky image sequence
        sky_seq = torch.stack([
            self.img_transform(Image.open(p).convert("RGB"))
            for p in img_window[self.sky_col].values
        ])

        # Extract normalized time-series inputs
        ts_seq = torch.tensor(
            ts_window[self.feature_cols].values,
            dtype=torch.float32
        )

        # Extract normalized target values
        target_seq = torch.tensor(
            target_window[self.target_cols].values,
            dtype=torch.float32
        )

        # Extract timestamps with second-level precision
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Path to the dataset CSV file
    CSV_PATH = "dataset_full_1M.csv"

    # Initialize dataset for inspection
    dataset = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        img_seq_len=5,
        ts_seq_len=30,
        horizon=25,
    )

    # Retrieve a single sample from the dataset
    sky_seq, ts_seq, target_seq, ts_time, tgt_time = dataset[2440]

    # Convert timestamp strings back to datetime objects
    ts_time = pd.to_datetime(ts_time)
    tgt_time = pd.to_datetime(tgt_time)

    # Print basic information about the sample
    print("Sky image sequence shape:", sky_seq.shape)
    print("Time-series input shape:", ts_seq.shape)
    print("Target sequence shape:", target_seq.shape)
    print("History time range:", ts_time[0], "→", ts_time[-1])
    print("Forecast time range:", tgt_time[0], "→", tgt_time[-1])

    # ------------------------------------------------------------
    # Plot sky image sequence
    # ------------------------------------------------------------
    T = sky_seq.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))
    if T == 1:
        axes = [axes]

    for i in range(T):
        img = sky_seq[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)

        axes[i].imshow(img)
        axes[i].set_title(ts_time[-T + i].strftime("%Y-%m-%d %H:%M:%S"))
        axes[i].axis("off")

    plt.suptitle("Sky Image Sequence")
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Plot time-series inputs and forecast targets
    # ------------------------------------------------------------

    # Convert normalized values back to original physical units
    mean = dataset.normalization_stats["mean"]
    std = dataset.normalization_stats["std"]

    ts_np = ts_seq.numpy() * std.values + mean.values
    tgt_np = (
        target_seq.numpy()
        * std[dataset.target_cols[0]]
        + mean[dataset.target_cols[0]]
    )

    # Combine history and forecast timestamps for x-axis labeling
    all_times = ts_time.append(tgt_time)
    all_labels = [t.strftime("%Y-%m-%d %H:%M:%S") for t in all_times]

    plt.figure(figsize=(16, 4))

    # Plot historical input features
    for i, col in enumerate(dataset.feature_cols):
        plt.plot(
            ts_time,
            ts_np[:, i],
            marker="o",
            label=col.upper()
        )

    # Plot forecast target values
    plt.plot(
        tgt_time,
        tgt_np[:, 0],
        "k--",
        marker="o",
        linewidth=2,
        label="GHI (Forecast)"
    )

    # Mark the boundary between history and forecast
    plt.axvline(ts_time[-1], color="gray", linestyle=":")

    # Display all timestamps on the x-axis
    plt.xticks(all_times, all_labels, rotation=90)

    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title("Time-Series Inputs and Forecast Target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
