# model_sky_predictor.py

import torch
import torch.nn as nn
from typing import Tuple, Optional


# ---------- Small CNN Encoder ----------
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 1/2
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, 3, stride=2, padding=1),   # 1/4
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


# ---------- CNN Decoder ----------
class ConvDecoder(nn.Module):
    def __init__(self, hidden_dim=64, out_channels=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # output scaled to [0,1]
        )

    def forward(self, x):
        return self.decoder(x)


# ---------- ConvLSTM Cell ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
            super().__init__()
            padding = kernel_size // 2
            self.hidden_dim = hidden_dim

            self.conv = nn.Conv2d(
                in_channels + hidden_dim,
                4 * hidden_dim,
                kernel_size,
                padding=padding,
            )

    def forward(self, x, state):
        h, c = state
        stacked = torch.cat([x, h], dim=1)
        gates = self.conv(stacked)

        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_state(self, batch, spatial):
        h, w = spatial
        device = next(self.parameters()).device
        return (
            torch.zeros(batch, self.hidden_dim, h, w, device=device),
            torch.zeros(batch, self.hidden_dim, h, w, device=device),
        )


# ---------- Sky Future Image Predictor ----------
class SkyFutureImagePredictor(nn.Module):
    """
    Input  : past image sequence  (B, Tin, C, H, W)
    Output : future image sequence (B, Tout, C, H, W)
    """

    def __init__(
        self,
        img_channels=3,
        hidden_dim=64,
        teacher_forcing=False,
    ):
        super().__init__()
        self.teacher_forcing = teacher_forcing

        self.encoder = ConvEncoder(img_channels, hidden_dim)
        self.convlstm = ConvLSTMCell(hidden_dim, hidden_dim)
        self.decoder = ConvDecoder(hidden_dim, img_channels)

    def forward(
        self,
        past_imgs: torch.Tensor,
        future_imgs: Optional[torch.Tensor] = None,
        horizon: int = 1,
    ):
        """
        past_imgs  : (B, Tin, C, H, W)
        future_imgs: (B, Tout, C, H, W) — only used for teacher forcing
        """

        B, Tin, C, H, W = past_imgs.shape

        # Encode past sequence
        features = []
        for t in range(Tin):
            f = self.encoder(past_imgs[:, t])
            features.append(f)

        Hf, Wf = features[0].shape[-2:]
        h, c = self.convlstm.init_state(B, (Hf, Wf))

        # Run ConvLSTM on past frames
        for f in features:
            h, c = self.convlstm(f, (h, c))

        preds = []
        x_t = features[-1]  # start decoding from last encoded state

        for t in range(horizon):
            # one ConvLSTM update step
            h, c = self.convlstm(x_t, (h, c))

            # decode to image
            out_img = self.decoder(h)
            preds.append(out_img)

            # autoregressive rollout OR teacher forcing
            if self.teacher_forcing and future_imgs is not None:
                x_t = self.encoder(future_imgs[:, t])
            else:
                x_t = self.encoder(out_img.detach())

        preds = torch.stack(preds, dim=1)  # (B, Tout, C, H, W)
        return preds
