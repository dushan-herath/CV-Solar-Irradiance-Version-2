import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

from dataset import IrradianceForecastDataset
from model_sky_predictor import SkyFutureImagePredictor

# ------------------------------
# ImageNet normalization values for sky images
# ------------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ------------------------------
# FutureImageWrapper for target sequences
# ------------------------------
class FutureImageWrapper(Dataset):
    def __init__(self, base_ds: IrradianceForecastDataset):
        self.ds = base_ds
        self.df = base_ds.df
        self.sky_col = base_ds.sky_col
        self.img_transform = base_ds.img_transform
        self.horizon = base_ds.horizon
        self.ts_seq_len = base_ds.ts_seq_len
        self.img_seq_len = base_ds.img_seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sky_seq, _, _, _, _ = self.ds[idx]

        # Future frames
        start = idx + self.ts_seq_len
        end = start + self.horizon
        future_df = self.df.iloc[start:end]
        future_imgs = torch.stack([
            self.img_transform(Image.open(p).convert("RGB"))
            for p in future_df[self.sky_col].values
        ])

        # Filenames for visualization
        past_names = self.df.iloc[idx:idx+self.img_seq_len][self.sky_col].tolist()
        future_names = future_df[self.sky_col].tolist()

        return sky_seq, future_imgs, past_names, future_names

# ------------------------------
# Device
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------
# Validation dataset parameters
# ------------------------------
CSV_PATH = "dataset_full_1M.csv"
IMG_SEQ_LEN = 5
TS_SEQ_LEN = 30
HORIZON = 5
IMG_SIZE = 64
FEATURE_COLS = ["ghi", "dni", "dhi"]

# ------------------------------
# Compute training statistics for numeric features
# ------------------------------
full_df = pd.read_csv(CSV_PATH)
train_df = full_df.iloc[:int(len(full_df)*0.8)]  # Assuming val_ratio=0.2
mean = train_df[FEATURE_COLS].mean()
std  = train_df[FEATURE_COLS].std()
normalization_stats = {"mean": mean, "std": std}

# ------------------------------
# Initialize validation dataset
# ------------------------------
val_base = IrradianceForecastDataset(
    csv_path=CSV_PATH,
    split="val",
    img_seq_len=IMG_SEQ_LEN,
    ts_seq_len=TS_SEQ_LEN,
    horizon=HORIZON,
    img_size=IMG_SIZE,
    normalization_stats=normalization_stats
)

val_ds = FutureImageWrapper(val_base)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

# ------------------------------
# Load trained model
# ------------------------------
model = SkyFutureImagePredictor(
    img_channels=3,
    hidden_dims=[64, 128, 256],
    lstm_hidden=[256, 256],
    teacher_forcing=False
).to(DEVICE)

model.load_state_dict(torch.load("best_sky_model.pth", map_location=DEVICE))
model.eval()
print("Loaded trained model.")

# ------------------------------
# Helper functions
# ------------------------------
def denormalize_image_tensor(img):
    """
    img: (C,H,W) tensor in ImageNet normalized space
    Returns: tensor in [0,1] image space
    """
    mean = torch.tensor(MEAN, device=img.device).view(-1,1,1)
    std  = torch.tensor(STD,  device=img.device).view(-1,1,1)
    img_denorm = img * std + mean
    return img_denorm.clamp(0,1)

def plot_sequence(past_imgs, pred_imgs, true_imgs, past_names, future_names, save_path=None):
    """
    Plots past, predicted, and true images (normalized and denormalized)
    """
    Tin = past_imgs.shape[0]
    Horizon = pred_imgs.shape[0]
    max_len = max(Tin, Horizon)

    fig, axs = plt.subplots(6, max_len, figsize=(3*max_len, 12))
    axs = np.atleast_2d(axs)

    # ------------------ Denormalize ------------------
    past_denorm = torch.stack([denormalize_image_tensor(img) for img in past_imgs])
    true_denorm = torch.stack([denormalize_image_tensor(img) for img in true_imgs])
    pred_denorm = torch.stack([denormalize_image_tensor(img) for img in pred_imgs])

    # Clamp normalized images for plotting
    past_norm = past_imgs.clone().clamp(0,1)
    pred_norm = pred_imgs.clone().clamp(0,1)
    true_norm = true_imgs.clone().clamp(0,1)

    # ------------------ Plot images ------------------
    for t in range(Tin):
        axs[0,t].imshow(past_norm[t].permute(1,2,0).cpu().numpy())
        axs[0,t].axis("off")
        axs[0,t].set_title(f"Past Norm {t+1}")

        axs[1,t].imshow(past_denorm[t].permute(1,2,0).cpu().numpy())
        axs[1,t].axis("off")
        axs[1,t].set_title(f"Past Denorm {t+1}")

    for t in range(Horizon):
        axs[2,t].imshow(pred_norm[t].permute(1,2,0).cpu().numpy())
        axs[2,t].axis("off")
        axs[2,t].set_title(f"Pred Norm {t+1}")

        axs[3,t].imshow(pred_denorm[t].permute(1,2,0).cpu().numpy())
        axs[3,t].axis("off")
        axs[3,t].set_title(f"Pred Denorm {t+1}")

        axs[4,t].imshow(true_norm[t].permute(1,2,0).cpu().numpy())
        axs[4,t].axis("off")
        axs[4,t].set_title(f"True Norm {t+1}")

        axs[5,t].imshow(true_denorm[t].permute(1,2,0).cpu().numpy())
        axs[5,t].axis("off")
        axs[5,t].set_title(f"True Denorm {t+1}")

    # Hide unused axes
    for row in range(6):
        for col in range(max_len):
            if (row < 2 and col >= Tin) or (row >=2 and col >= Horizon):
                axs[row,col].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    plt.show()

# ------------------------------
# Visualize some validation samples
# ------------------------------
NUM_SAMPLES = 5
with torch.no_grad():
    for i, (past_imgs, future_imgs, past_names, future_names) in enumerate(val_loader):
        past_imgs = past_imgs.to(DEVICE)
        future_imgs = future_imgs.to(DEVICE)

        preds = model(past_imgs, horizon=HORIZON)

        # Remove batch dimension
        past_imgs = past_imgs[0]
        future_imgs = future_imgs[0]
        preds = preds[0]

        plot_sequence(past_imgs, preds, future_imgs, past_names[0], future_names[0])

        if i+1 >= NUM_SAMPLES:
            break
