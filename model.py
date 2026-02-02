import math
import torch
from torch import nn
import timm

# ----------------------
# Utility: causal mask
# ----------------------
def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

# ----------------------
# Vision Transformer Encoder
# ----------------------
class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        model_name="vit_base_patch16_224",
        img_size=224,
        pretrained=True,
        freeze=True,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            global_pool="avg",
        )

        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

# ----------------------
# Time-series encoder
# ----------------------
class TS_Encoder(nn.Module):
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
        return self.net(x)

# ----------------------
# Positional encoding
# ----------------------
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

# ----------------------
# Weighted Gated Fusion (Option 1)
# ----------------------
class WeightedGatedFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, out_dim):
        super().__init__()

        self.img_proj = nn.Linear(img_dim, ts_dim)

        self.weight_net = nn.Sequential(
            nn.Linear(ts_dim * 2, ts_dim),
            nn.GELU(),
            nn.Linear(ts_dim, ts_dim),
            nn.Sigmoid(),   # α ∈ [0,1]
        )

        self.proj = nn.Sequential(
            nn.Linear(ts_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, ts, img):
        img = self.img_proj(img)

        alpha = self.weight_net(torch.cat([ts, img], dim=-1))  # shape: (B, T, ts_dim)

        fused = alpha * ts + (1.0 - alpha) * img
        return self.proj(fused)

# ----------------------
# Temporal transformer (pure)
# ----------------------
class TemporalTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        mask = causal_mask(x.size(1), x.device)
        return self.encoder(x, mask)

# ----------------------
# Multimodal forecaster
# ----------------------
class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        sky_encoder,
        ts_feat_dim,
        ts_embed_dim=64,
        fused_dim=128,
        horizon=25,
        target_dim=1,
    ):
        super().__init__()

        self.sky_encoder = sky_encoder
        self.ts_encoder = TS_Encoder(ts_feat_dim, ts_embed_dim)

        self.fusion = WeightedGatedFusion(
            img_dim=sky_encoder.out_dim,
            ts_dim=ts_embed_dim,
            out_dim=fused_dim,
        )

        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(fused_dim)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, horizon * target_dim),
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky, ts):
        """
        sky : (B, T_img, C, H, W)
        ts  : (B, T_ts, ts_feat_dim)
        """
        B, T_img = sky.shape[:2]

        # Encode sky images
        B, T, C, H, W = sky.shape
        sky = self.sky_encoder(
            sky.view(B * T, C, H, W)
        ).view(B, T, -1)

        # Encode time-series
        ts = self.ts_encoder(ts)

        # Align TS with image frames
        ts_img = ts[:, -T_img:]

        # Weighted gated fusion
        fused = self.fusion(ts_img, sky)

        # Temporal modeling
        fused = self.pos_enc(fused)
        fused = self.temporal_tf(fused)

        # Forecast from last timestep
        context = fused[:, -1]
        out = self.head(context)

        return out.view(B, self.horizon, self.target_dim)

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sky_enc = VisionTransformerEncoder(
        model_name="vit_base_patch16_224",
        img_size=224,
        pretrained=True,
        freeze=True,
    )

    model = MultimodalForecaster(
        sky_encoder=sky_enc,
        ts_feat_dim=3,
        horizon=25,
        target_dim=1,
    ).to(device)

    B, T_img, T_ts = 2, 5, 30
    sky = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 3).to(device)

    y = model(sky, ts)
    print("Output shape:", y.shape)
