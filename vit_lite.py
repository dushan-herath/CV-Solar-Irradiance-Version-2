import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================
# Utility: Fisheye Geometry Map (radius + hemisphere height)
# ==================================================
def fisheye_map(H, W, device):
    """
    Returns a 2-channel fisheye map:
    - Channel 0: radial distance r (0=center, 1=edge)
    - Channel 1: height z of hemisphere (0=edge, 1=center)
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    r = torch.sqrt(x**2 + y**2)
    z = torch.sqrt(torch.clamp(1.0 - r**2, min=0.0))  # hemisphere height
    return torch.stack([r, z], dim=0)  # (2, H, W)


def patch_radii(num_patches, device):
    size = int(num_patches ** 0.5)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, size, device=device),
        torch.linspace(-1, 1, size, device=device),
        indexing="ij"
    )
    r = torch.sqrt(x ** 2 + y ** 2)
    return r.flatten()


# ==================================================
# Patch Embedding (Conv Stem)
# ==================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=5, embed_dim=128):
        super().__init__()
        assert patch_size in [4, 8, 16, 32], "patch_size must be one of [4, 8, 16, 32]"

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),

            nn.Conv2d(embed_dim // 2, embed_dim // 2, 3, 2, 1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),

            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x  # (B, N, D)


# ==================================================
# Transformer Block
# ==================================================
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
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            need_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ==================================================
# ViT-Lite (Fisheye-aware, RGB + geometry)
# ==================================================
class ViTLite(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.2,
        use_geometry=False  # <-- toggle r,z channels
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_dim = embed_dim
        self.use_geometry = use_geometry

        in_chans = 3 + 2 if use_geometry else 3  # RGB + optional r,z

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        B, C, H, W = x.shape

        # ------------------------------------------------
        # Optionally add fisheye geometry
        # ------------------------------------------------
        if self.use_geometry:
            geo = fisheye_map(H, W, x.device)      # (2, H, W)
            geo = geo.unsqueeze(0).expand(B, -1, H, W)
            x = torch.cat([x, geo], dim=1)         # (B, 5, H, W)

        # ------------------------------------------------
        # Patch embedding
        # ------------------------------------------------
        x = self.patch_embed(x)  # (B, N, D)

        # ------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # ------------------------------------------------
        # Pooling (mean over patches)
        # ------------------------------------------------
        x = x.mean(dim=1)  # (B, D)

        return x




# ==================================================
# Sanity Test
# ==================================================
if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)  # Batch of RGB images

    model = ViTLite(
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4
    )

    y = model(x)
    print("Output shape:", y.shape)  # (2, 128)
    print("out_dim:", model.out_dim)
