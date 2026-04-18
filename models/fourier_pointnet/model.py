import math
import os

import torch
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential


class FourierPointNet(Module):
    """VanillaPointNet + Fourier positional encoding.

    Per-point input: pos (3) + fourier(pos) (3 * 2 * F) + velocity_in (15) + t_start (1).
    With F=6 frequencies [1,2,4,8,16,32], input dim = 3 + 36 + 15 + 1 = 55.
    """

    FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "freqs",
            2.0 * math.pi * torch.tensor(self.FREQS, dtype=torch.float32),
        )
        in_dim = 3 + 3 * 2 * len(self.FREQS) + 15 + 1

        self.encoder = Sequential(
            Linear(in_dim, 128),
            LayerNorm(128),
            ReLU(),
            Linear(128, 128),
            LayerNorm(128),
            ReLU(),
        )

        self.decoder = Sequential(
            Linear(256, 256),
            LayerNorm(256),
            ReLU(),
            Linear(256, 15),
        )

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, N, 3) -> (B, N, 3*2*F)
        # angles: (B, N, 3, F)
        angles = pos.unsqueeze(-1) * self.freqs
        sin = angles.sin()
        cos = angles.cos()
        feats = torch.stack([sin, cos], dim=-1)  # (B, N, 3, F, 2)
        return feats.flatten(start_dim=2)

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape

        vel_flat = velocity_in.transpose(1, 2).reshape(batch_size, num_pos, num_t_in * 3)
        t_start = t[:, 0:1].unsqueeze(1).expand(-1, num_pos, -1)
        pos_fourier = self._fourier(pos)

        x = torch.cat([pos, pos_fourier, vel_flat, t_start], dim=2)

        local_feat = self.encoder(x)
        global_feat = local_feat.max(dim=1).values
        global_feat = global_feat.unsqueeze(1).expand(-1, num_pos, -1)
        combined = torch.cat([local_feat, global_feat], dim=2)

        delta = self.decoder(combined).view(batch_size, num_pos, num_t_in, 3)
        last_frame = velocity_in[:, -1, :, :]
        out = last_frame.unsqueeze(2) + delta

        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2)
