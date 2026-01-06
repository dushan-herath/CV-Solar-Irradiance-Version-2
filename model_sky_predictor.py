import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ----------------------
# Conv Encoder (U-Net style)
# ----------------------
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256]):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for h in hidden_dims:
            conv = nn.Sequential(
                nn.Conv2d(prev_ch, h, 3, stride=2, padding=1),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True)
            )
            layers.append(conv)
            prev_ch = h
        self.layers = nn.ModuleList(layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x):
        skips = []
        for conv in self.layers:
            x = conv(x)
            skips.append(x)
        return x, skips  # bottleneck + skip features


# ----------------------
# Conv Decoder with skip connections
# ----------------------
class ConvDecoder(nn.Module):
    def __init__(self, hidden_dims=[256, 128, 64], out_channels=3, encoder_dims=[64,128,256]):
        super().__init__()
        self.encoder_dims = encoder_dims  # keep in original order
        layers = []
        self.skip_projs = nn.ModuleList()

        for i, h in enumerate(hidden_dims):
            out_ch = hidden_dims[i+1] if i+1 < len(hidden_dims) else hidden_dims[i]
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(h, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))

            # projection for skip connection if channels differ
            skip_ch = self.encoder_dims[i] if i < len(self.encoder_dims) else 0
            if skip_ch != out_ch:
                self.skip_projs.append(nn.Conv2d(skip_ch, out_ch, 1))
            else:
                self.skip_projs.append(nn.Identity())

        self.layers = nn.ModuleList(layers)
        self.final = nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1)  # no Sigmoid

    def forward(self, x, skips=None):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if skips is not None and i < len(skips):
                skip = skips[i]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                skip = self.skip_projs[i](skip)
                x = x + skip
        x = self.final(x)
        return x



# ----------------------
# ConvLSTM Cell
# ----------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels + hidden_dim, 4*hidden_dim, kernel_size, padding=padding)

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
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


# ----------------------
# Stacked ConvLSTM
# ----------------------
class StackedConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dims=[256, 256]):
        super().__init__()
        self.cells = nn.ModuleList()
        for i, h in enumerate(hidden_dims):
            ch_in = in_channels if i==0 else hidden_dims[i-1]
            self.cells.append(ConvLSTMCell(ch_in, h))
        self.hidden_dims = hidden_dims

    def forward(self, x, states=None):
        if states is None:
            states = [cell.init_state(x.size(0), x.shape[-2:]) for cell in self.cells]

        new_states = []
        h = x
        for i, cell in enumerate(self.cells):
            h, c = cell(h, states[i])
            new_states.append((h, c))
        return h, new_states


# ----------------------
# Sky Future Image Predictor
# ----------------------
class SkyFutureImagePredictor(nn.Module):
    """
    ConvLSTM + U-Net style encoder-decoder for sky image prediction
    """
    def __init__(self,
                 img_channels=3,
                 hidden_dims=[64,128,256],
                 lstm_hidden=[256,256],
                 teacher_forcing=False):
        super().__init__()
        self.teacher_forcing = teacher_forcing

        self.encoder = ConvEncoder(img_channels, hidden_dims)
        self.lstm = StackedConvLSTM(hidden_dims[-1], lstm_hidden)
        self.decoder = ConvDecoder(hidden_dims[::-1], img_channels, encoder_dims=hidden_dims)

    def forward(self, past_imgs, future_imgs: Optional[torch.Tensor]=None, horizon:int=1):
        B, Tin, C, H, W = past_imgs.shape
        skip_feats = []

        # Encode past frames
        features = []
        for t in range(Tin):
            f, skips = self.encoder(past_imgs[:,t])
            features.append(f)
            skip_feats.append(skips)

        # Initialize ConvLSTM states
        h, w = features[0].shape[-2:]
        states = [cell.init_state(B, (h,w)) for cell in self.lstm.cells]

        # Step through past sequence
        for f in features:
            out, states = self.lstm(f, states)

        preds = []
        x_t = out  # start with last encoded feature

        # Prediction loop
        for t in range(horizon):
            out, states = self.lstm(x_t, states)
            # Use last skip features for decoder (or could experiment with averaging)
            pred_img = self.decoder(out, skips=skip_feats[-1])
            preds.append(pred_img)

            # Teacher forcing / autoregressive
            if self.teacher_forcing and future_imgs is not None:
                x_t, _ = self.encoder(future_imgs[:, t])
            else:
                x_t, _ = self.encoder(pred_img)  # keep gradient for learning

        preds = torch.stack(preds, dim=1)  # (B, horizon, C, H, W)
        return preds
