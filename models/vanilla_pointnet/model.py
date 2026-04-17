import os

import torch
from torch.nn import LayerNorm, Linear, Module, ReLU, Sequential


class VanillaPointNet(Module):
    """PointNet-style model: per-point encoder, global max-pool, per-point decoder.

    Input per point: pos (3) + velocity_in flattened (15) + t_start (1) = 19.
    Encoder lifts to local features, max-pool gives a global geometry summary,
    decoder predicts per-point velocity deltas relative to the last input frame.
    Airfoil points are hard-masked to zero velocity in the output.
    """

    def __init__(self):
        super().__init__()

        self.encoder = Sequential(
            Linear(19, 128),
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

        # Load weights if available
        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_t_in, num_pos, _ = velocity_in.shape

        # Flatten velocity input: (B, 5, N, 3) -> (B, N, 15)
        vel_flat = velocity_in.transpose(1, 2).reshape(batch_size, num_pos, num_t_in * 3)

        # Start time as a per-point scalar: (B, 1) -> (B, N, 1)
        t_start = t[:, 0:1].unsqueeze(1).expand(-1, num_pos, -1)

        # Per-point input: (B, N, 19)
        x = torch.cat([pos, vel_flat, t_start], dim=2)

        # Encode: (B, N, 128)
        local_feat = self.encoder(x)

        # Global max-pool: (B, 128)
        global_feat = local_feat.max(dim=1).values

        # Broadcast and concat: (B, N, 256)
        global_feat = global_feat.unsqueeze(1).expand(-1, num_pos, -1)
        combined = torch.cat([local_feat, global_feat], dim=2)

        # Decode deltas: (B, N, 15)
        delta = self.decoder(combined)

        # Reshape to (B, N, 5, 3), add to last input frame
        delta = delta.view(batch_size, num_pos, num_t_in, 3)
        last_frame = velocity_in[:, -1, :, :]  # (B, N, 3)
        out = last_frame.unsqueeze(2) + delta    # (B, N, 5, 3)

        # Hard-mask airfoil surface points to zero (no-slip)
        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        # Reshape to expected output: (B, 5, N, 3)
        out = out.transpose(1, 2)

        return out
