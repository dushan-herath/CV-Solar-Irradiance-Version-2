import os
import torch
import contextlib
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IrradianceForecastDataset
from model_sky_predictor import SkyFutureImagePredictor
from model_time_series_predictor import TimeSeriesForecaster

# -------------------------
# Dataset wrapper
# -------------------------
class CombinedGHI_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_ds: IrradianceForecastDataset):
        self.ds = base_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sky_seq, ts_past, ghi_future, _, _ = self.ds[idx]
        return sky_seq, ts_past.float(), ghi_future.float()


# -------------------------
# Combined model (reg head built externally)
# -------------------------
class SkyTS_GHI_Model(nn.Module):
    def __init__(self, sky_model: SkyFutureImagePredictor, ts_model: TimeSeriesForecaster, reg_input_dim: int, reg_hidden=128, horizon=25):
        """
        reg_input_dim: actual concatenated feature size (sky + ts)
        """
        super().__init__()
        self.sky_model = sky_model
        self.ts_model = ts_model
        self.horizon = horizon

        # Freeze pretrained models
        for p in self.sky_model.parameters():
            p.requires_grad = False
        for p in self.ts_model.parameters():
            p.requires_grad = False

        # Regression head (built with actual input dim)
        self.reg_head = nn.Sequential(
            nn.Linear(reg_input_dim, reg_hidden),
            nn.ReLU(),
            nn.Linear(reg_hidden, 1)   # predicting first horizon step (change if you want full horizon)
        )

    def forward(self, past_imgs, past_ts):
        """
        past_imgs: [B, T_img, C, H, W]
        past_ts:   [B, T_ts, D]
        """
        B, T_img, C, H, W = past_imgs.shape

        # --- sky features ---
        with torch.no_grad():
            x = past_imgs.view(B * T_img, C, H, W)
            sky_feat = self.sky_model.encoder(x)   # maybe tensor or (tensor, other)
            if isinstance(sky_feat, tuple):
                sky_feat = sky_feat[0]

            # If returned tensor is 4D [B*T, C, H', W'] -> global avg pool
            if sky_feat.dim() == 4:
                sky_feat = torch.mean(sky_feat, dim=[2, 3])  # [B*T, C]
            # now ensure [B*T, C] -> [B, T, C]
            sky_feat = sky_feat.view(B, T_img, -1)
            sky_latent = sky_feat[:, -1, :]  # use last timestep -> [B, sky_C]

        # --- ts features ---
        with torch.no_grad():
            ts_out = self.ts_model(past_ts)  # could be [B, horizon, 1] or [B, horizon] or [B, feat]
            if ts_out.dim() == 3:
                ts_latent = ts_out.view(B, -1)   # flatten to [B, horizon*1]
            elif ts_out.dim() == 2:
                ts_latent = ts_out               # already [B, N]
            else:
                # fallback: flatten any other shape
                ts_latent = ts_out.view(B, -1)

        # --- concat & regress ---
        combined = torch.cat([sky_latent, ts_latent], dim=-1)  # [B, reg_input_dim]
        ghi_pred = self.reg_head(combined)                     # [B, 1]
        return ghi_pred


# -------------------------
# Training / validation
# -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc="Training")
    for past_imgs, past_ts, future_ghi in loop:
        past_imgs = past_imgs.to(device)
        past_ts = past_ts.to(device)
        future_ghi = future_ghi.to(device)

        optimizer.zero_grad()
        ac = torch.cuda.amp.autocast() if device.type == "cuda" else contextlib.nullcontext()
        with ac:
            preds = model(past_imgs, past_ts)           # [B, 1]
            loss = criterion(preds, future_ghi[:, 0:1]) # first horizon step

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        loop.set_postfix({"loss": total_loss / (loop.n + 1)})
    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        loop = tqdm(loader, desc="Validation")
        for past_imgs, past_ts, future_ghi in loop:
            past_imgs = past_imgs.to(device)
            past_ts = past_ts.to(device)
            future_ghi = future_ghi.to(device)

            preds = model(past_imgs, past_ts)
            loss = criterion(preds, future_ghi[:, 0:1])
            total_loss += loss.item()
            loop.set_postfix({"val_loss": total_loss / (loop.n + 1)})
    return total_loss / len(loader)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    CSV_PATH = "dataset_full_1M.csv"
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    IMG_SEQ_LEN = 5
    TS_SEQ_LEN = 30
    HORIZON = 25
    IMG_SIZE = 64
    TS_FEAT_DIM = 3

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE}")

    # Dataset
    train_base = IrradianceForecastDataset(CSV_PATH, split="train",
                                           img_seq_len=IMG_SEQ_LEN,
                                           ts_seq_len=TS_SEQ_LEN,
                                           horizon=HORIZON,
                                           img_size=IMG_SIZE)
    val_base = IrradianceForecastDataset(CSV_PATH, split="val",
                                         img_seq_len=IMG_SEQ_LEN,
                                         ts_seq_len=TS_SEQ_LEN,
                                         horizon=HORIZON,
                                         img_size=IMG_SIZE,
                                         normalization_stats=train_base.normalization_stats)

    train_ds = CombinedGHI_Dataset(train_base)
    val_ds = CombinedGHI_Dataset(val_base)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Load pretrained models
    sky_model = SkyFutureImagePredictor(img_channels=3,
                                        hidden_dims=[64,128,256],
                                        lstm_hidden=[256,256],
                                        teacher_forcing=True).to(DEVICE)
    sky_model.load_state_dict(torch.load("best_sky_model.pth", map_location=DEVICE))

    ts_model = TimeSeriesForecaster(ts_feat_dim=TS_FEAT_DIM,
                                    ts_embed_dim=64,
                                    model_dim=128,
                                    horizon=HORIZON,
                                    target_dim=1).to(DEVICE)
    ts_model.load_state_dict(torch.load("best_ts_model.pth", map_location=DEVICE))

    # --- compute actual reg_input_dim using dummy batch ---
    with torch.no_grad():
        dummy_img = torch.zeros(1, IMG_SEQ_LEN, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        sky_feat = sky_model.encoder(dummy_img.view(1 * IMG_SEQ_LEN, 3, IMG_SIZE, IMG_SIZE))
        if isinstance(sky_feat, tuple):
            sky_feat = sky_feat[0]
        if sky_feat.dim() == 4:
            sky_feat = torch.mean(sky_feat, dim=[2, 3])  # [1*T, C]
        sky_feat = sky_feat.view(1, IMG_SEQ_LEN, -1)
        sky_latent = sky_feat[:, -1, :]              # [1, sky_C]
        sky_latent_dim = sky_latent.shape[-1]

        dummy_ts = torch.zeros(1, TS_SEQ_LEN, TS_FEAT_DIM).to(DEVICE)
        ts_out = ts_model(dummy_ts)
        if ts_out.dim() == 3:
            ts_latent_dim = ts_out.view(1, -1).shape[-1]
        else:
            ts_latent_dim = ts_out.shape[-1]

    reg_input_dim = sky_latent_dim + ts_latent_dim
    print(f"Computed reg_input_dim = {reg_input_dim} (sky {sky_latent_dim} + ts {ts_latent_dim})")

    # Instantiate combined model with correct reg_input_dim
    model = SkyTS_GHI_Model(sky_model, ts_model, reg_input_dim=reg_input_dim, horizon=HORIZON).to(DEVICE)

    # Only train reg_head parameters
    optimizer = torch.optim.AdamW(model.reg_head.parameters(), lr=4e-4, weight_decay=1e-4)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_combined_ghi_model.pth")
            print("Saved best model")
