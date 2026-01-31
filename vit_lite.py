import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================
# Utility: Resize Positional Embeddings
# ==================================================
def resize_pos_embed(pos_embed, new_num_patches):
    """
    Dynamically resize positional embeddings if image resolution changes.
    """
    _, N, D = pos_embed.shape
    if N == new_num_patches:
        return pos_embed

    orig_size = int(N ** 0.5)
    new_size = int(new_num_patches ** 0.5)

    pos = pos_embed.reshape(1, orig_size, orig_size, D).permute(0, 3, 1, 2)
    pos = F.interpolate(
        pos,
        size=(new_size, new_size),
        mode="bilinear",
        align_corners=False
    )
    pos = pos.permute(0, 2, 3, 1).reshape(1, new_num_patches, D)
    return pos


# ==================================================
# Patch Embedding (Conv Stem)
# ==================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        assert patch_size in [4, 8, 16], "Use powers of 2 for conv stem"

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),

            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),

            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


# ==================================================
# Transformer Block
# ==================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True
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
# ViT-Lite (Basic Vision Transformer)
# ==================================================
class ViTLite(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.2
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

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
        # ----------------------------------------------
        # 1. Patch embedding
        # ----------------------------------------------
        x = self.patch_embed(x)  # (B, N, D)
        N = x.shape[1]

        # ----------------------------------------------
        # 2. Positional embedding
        # ----------------------------------------------
        pos_embed = resize_pos_embed(self.pos_embed, N).to(x.device)
        x = x + pos_embed
        x = self.pos_drop(x)

        # ----------------------------------------------
        # 3. Transformer encoder
        # ----------------------------------------------
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # ----------------------------------------------
        # 4. Global average pooling
        # ----------------------------------------------
        return x.mean(dim=1)  # (B, D)


# ==================================================
# Sanity Test
# ==================================================
if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)

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
