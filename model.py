import math
import torch
from torch import nn
import timm


def causal_mask(T, device):
    # Prevents the model from attending to future time steps
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


class TemporalConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Lightweight temporal convolution for local sequence refinement
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class ImageEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, freeze=True):
        super().__init__()
        # CNN backbone used to extract features from sky images
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class TS_Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        # Projects raw time-series features into a shared embedding space
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Standard sinusoidal positional encoding
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CrossModalFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, out_dim):
        super().__init__()
        # Aligns image features to the time-series embedding space
        self.img_proj = nn.Linear(img_dim, ts_dim)

        # Cross-attention where TS queries attend to sky features
        self.attn = nn.MultiheadAttention(
            ts_dim, num_heads=4, batch_first=True
        )

        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, ts, img):
        img = self.img_proj(img)
        attn, _ = self.attn(ts, img, img)
        return self.proj(torch.cat([ts, attn], dim=-1))


class TemporalTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Causal transformer encoder for temporal modeling
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

        self.cross_fusion = CrossModalFusion(
            img_dim=sky_encoder.out_dim,
            ts_dim=ts_embed_dim,
            out_dim=fused_dim,
        )

        self.temp_conv = TemporalConvBlock(fused_dim)
        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(fused_dim)

        # Final regression head for multi-step forecasting
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, horizon * target_dim),
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky, ts):
        B, T_img = sky.shape[:2]

        # Encode each sky image independently using the CNN
        B, T, C, H, W = sky.shape
        sky = self.sky_encoder(
            sky.view(B * T, C, H, W)
        ).view(B, T, -1)

        # Encode the time-series measurements
        ts = self.ts_encoder(ts)

        # Match the last image frames with the most recent TS steps
        ts_img = ts[:, -T_img:]

        fused = self.cross_fusion(ts_img, sky)
        fused = fused + self.temp_conv(fused)

        fused = self.pos_enc(fused)
        fused = self.temporal_tf(fused)

        # Use the final time step as the forecasting context
        context = fused[:, -1]
        out = self.head(context)

        return out.view(B, self.horizon, self.target_dim)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sky_enc = ImageEncoder("resnet18", pretrained=True)

    model = MultimodalForecaster(
        sky_encoder=sky_enc,
        ts_feat_dim=3,
    ).to(device)

    B, T_img, T_ts = 2, 5, 30
    sky = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 3).to(device)

    y = model(sky, ts)
    print("Output shape:", y.shape)
