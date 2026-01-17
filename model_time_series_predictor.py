import math
import torch
from torch import nn


# ============================================================
# Utilities
# ============================================================
def causal_mask(T, device):
    """
    Prevents attention to future time steps
    """
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


# ============================================================
# Temporal Convolution Block
# ============================================================
class TemporalConvBlock(nn.Module):
    """
    Lightweight depthwise temporal convolution
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, T, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


# ============================================================
# Time-Series Encoder
# ============================================================
class TS_Encoder(nn.Module):
    """
    Projects raw TS features into embedding space
    """
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x: (B, T, D)
        return self.net(x)


# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ============================================================
# Temporal Transformer (Causal)
# ============================================================
class TemporalTransformer(nn.Module):
    """
    Causal Transformer encoder for TS modeling
    """
    def __init__(self, d_model, nhead=8, num_layers=3):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, T, D)
        mask = causal_mask(x.size(1), x.device)
        return self.encoder(x, mask)


# ============================================================
# Time-Series GHI Forecaster (Transformer-based)
# ============================================================
class TimeSeriesForecaster(nn.Module):
    """
    Transformer-based time-series forecaster

    Inputs (per timestep):
        [ghi, dni, dhi, temp, pressure,
         tod_sin, tod_cos, doy_sin, doy_cos,
         solar_zenith, solar_azimuth]

    Output:
        Future GHI
    """
    def __init__(
        self,
        ts_feat_dim,
        ts_embed_dim=64,
        model_dim=128,
        horizon=25,
        target_dim=1,
    ):
        super().__init__()

        self.ts_encoder = TS_Encoder(ts_feat_dim, ts_embed_dim)

        self.input_proj = nn.Linear(ts_embed_dim, model_dim)

        self.temp_conv = TemporalConvBlock(model_dim)
        self.pos_enc = PositionalEncoding(model_dim)
        self.temporal_tf = TemporalTransformer(model_dim)

        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, horizon * target_dim),
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, ts):
        """
        ts: (B, T, D)
        return: (B, horizon, target_dim)
        """
        # Encode raw features
        x = self.ts_encoder(ts)          # (B, T, E)
        x = self.input_proj(x)           # (B, T, D_model)

        # Local temporal refinement
        x = x + self.temp_conv(x)

        # Positional + transformer
        x = self.pos_enc(x)
        x = self.temporal_tf(x)

        # Use last timestep as context
        context = x[:, -1]

        out = self.head(context)
        return out.view(ts.size(0), self.horizon, self.target_dim)


# ============================================================
# Sanity Check
# ============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    T = 60
    D = 11   # ghi, dni, dhi, temp, pressure, etc.

    model = TimeSeriesForecaster(
        ts_feat_dim=D,
        ts_embed_dim=64,
        model_dim=128,
        horizon=25,
        target_dim=1,
    ).to(device)

    ts = torch.randn(B, T, D).to(device)
    y = model(ts)

    print("Output shape:", y.shape)  # (B, 25, 1)
