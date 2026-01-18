import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==================================================
# ------------------ ViT-Lite ---------------------
# ==================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                     # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)     # (B, N, D)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ViTLite(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=128, depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.2):
        super().__init__()
        self.out_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)  # Global average pooling

# ==================================================
# ---------------- SESIB Components ----------------
# ==================================================
class SolarProxyHead(nn.Module):
    """
    Learns a soft solar-importance map from the image itself.
    """
    def __init__(self, in_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)  # [B, 1, H, W]

class SESIBViTWrapper(nn.Module):
    """
    Wraps ViTLite with SESIB additive solar bias.
    """
    def __init__(self, vit_model: nn.Module, patch_size=8):
        super().__init__()
        self.vit = vit_model
        self.patch_size = patch_size
        self.solar_proxy = SolarProxyHead(in_ch=3)
        self.lambda_phys = nn.Parameter(torch.tensor(0.1))

        self.out_dim = vit_model.out_dim

    def forward(self, x):
        B, C, H, W = x.shape

        # Solar proxy map
        solar_map = self.solar_proxy(x)           # [B, 1, H, W]
        solar_bias = F.avg_pool2d(
            solar_map, kernel_size=self.patch_size, stride=self.patch_size
        ).flatten(1)                              # [B, N_patches]

        # Patch embeddings
        x_patch = self.vit.patch_embed(x)
        x_patch = x_patch + self.vit.pos_embed
        x_patch = x_patch + self.vit.pos_drop(x_patch)

        # Add patch-level bias to embeddings before attention
        x_patch = x_patch + self.lambda_phys * solar_bias.unsqueeze(-1)  # [B, N, D]

        # Forward through blocks (unchanged)
        for blk in self.vit.blocks:
            x_patch = blk(x_patch)

        x_patch = self.vit.norm(x_patch)
        return x_patch.mean(dim=1)


# ==================================================
# ------------------- Test Run --------------------
# ==================================================
if __name__ == "__main__":
    # Dummy batch of sky images
    x = torch.rand(2, 3, 64, 64)  # [B, C, H, W]

    vit_model = ViTLite(
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.2
    )

    model = SESIBViTWrapper(vit_model, patch_size=8)
    out = model(x)
    print("Output shape:", out.shape)  # [B, embed_dim]

    # Optional: visualize solar proxy map
    with torch.no_grad():
        solar_map = model.solar_proxy(x)
        print("Solar map shape:", solar_map.shape)  # [B, 1, H, W]
