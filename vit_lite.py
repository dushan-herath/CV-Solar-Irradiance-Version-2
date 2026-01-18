import torch
import torch.nn as nn
import math
from einops import rearrange


# -------------------------------------------------
# Patch Embedding
# -------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                     # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)     # (B, N, D)
        return x


# -------------------------------------------------
# Transformer Block
# -------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads,
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
        # Self-attention
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            need_weights=False
        )
        x = x + attn_out

        # Feedforward
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------------------------
# ViT-Lite Encoder
# -------------------------------------------------
class ViTLite(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=8,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.2
    ):
        super().__init__()

        self.out_dim = embed_dim
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
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
        """
        x: (B, C, H, W)
        returns: (B, D)
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # 🔑 Mean pooling instead of CLS
        x = x.mean(dim=1)
        return x
