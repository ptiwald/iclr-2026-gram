import math
import os

import torch
from torch.nn import LayerNorm, Linear, Module, ModuleList, ReLU, Sequential
from torch_cluster import knn


class KNNPointNet(Module):
    """PointNet + local k-NN message passing.

    Per-point input: pos (3) + fourier(pos) (3 * 2 * F) + velocity_in (15) + t_start (1).
    With F=6 frequencies [1,2,4,8,16,32], input dim = 3 + 36 + 15 + 1 = 55.

    Step 1: per-point encoder -> 128.
    Step 2: `num_rounds` rounds of k-NN message passing. Each round gathers k neighbors'
            features, max-pools, concatenates with the point's own features, and maps
            256 -> 128 via an MLP with independent weights per round.
    Step 3: global max-pool -> 128; broadcast and concat with local features -> 256.
    Step 4: per-point decoder 256 -> 15 (5 frames x 3 delta components).
    Step 5: add deltas to last input frame; hard-mask airfoil points to zero.
    """

    FREQS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

    def __init__(self, num_rounds: int = 2, k: int = 16, hidden: int = 128):
        super().__init__()

        self.num_rounds = num_rounds
        self.k = k

        self.register_buffer(
            "freqs",
            2.0 * math.pi * torch.tensor(self.FREQS, dtype=torch.float32),
        )
        in_dim = 3 + 3 * 2 * len(self.FREQS) + 15 + 1

        self.encoder = Sequential(
            Linear(in_dim, hidden),
            LayerNorm(hidden),
            ReLU(),
            Linear(hidden, hidden),
            LayerNorm(hidden),
            ReLU(),
        )

        self.mp_mlps = ModuleList([
            Sequential(
                Linear(2 * hidden, hidden),
                LayerNorm(hidden),
                ReLU(),
                Linear(hidden, hidden),
                LayerNorm(hidden),
                ReLU(),
            )
            for _ in range(num_rounds)
        ])

        self.decoder = Sequential(
            Linear(2 * hidden, 2 * hidden),
            LayerNorm(2 * hidden),
            ReLU(),
            Linear(2 * hidden, 15),
        )

        path = os.path.join(os.path.dirname(__file__), "state_dict.pt")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))

    def _fourier(self, pos: torch.Tensor) -> torch.Tensor:
        angles = pos.unsqueeze(-1) * self.freqs
        feats = torch.stack([angles.sin(), angles.cos()], dim=-1)  # (B, N, 3, F, 2)
        return feats.flatten(start_dim=2)

    def _knn_maxpool(self, feats_flat: torch.Tensor, pos_flat: torch.Tensor,
                     batch: torch.Tensor) -> torch.Tensor:
        """Max-pool over k nearest neighbors for each point.

        feats_flat: (B*N, D), pos_flat: (B*N, 3), batch: (B*N,) -> (B*N, D).
        """
        # knn(x, y, k, batch_x, batch_y): for each point in y, finds k nearest in x.
        # Returns edge_index (2, k*|y|): row 0 = query idx (in y), row 1 = neighbor idx (in x).
        # Ordering: edges are grouped by query, so reshaping to (|y|, k) is safe.
        edge = knn(pos_flat, pos_flat, self.k, batch_x=batch, batch_y=batch)
        n_total = feats_flat.shape[0]
        neigh_idx = edge[1].view(n_total, self.k)
        neigh = feats_flat[neigh_idx]  # (B*N, k, D)
        return neigh.max(dim=1).values

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

        x = torch.cat([pos, pos_fourier, vel_flat, t_start], dim=2)  # (B, N, in_dim)

        local = self.encoder(x)  # (B, N, H)

        # Flatten for k-NN. batch vector marks which sample each point belongs to.
        local_flat = local.reshape(batch_size * num_pos, -1)
        pos_flat = pos.reshape(batch_size * num_pos, 3)
        batch_vec = torch.arange(batch_size, device=pos.device).repeat_interleave(num_pos)

        for mlp in self.mp_mlps:
            neigh = self._knn_maxpool(local_flat, pos_flat, batch_vec)
            local_flat = mlp(torch.cat([local_flat, neigh], dim=-1))

        local = local_flat.view(batch_size, num_pos, -1)  # (B, N, H)

        global_feat = local.max(dim=1).values  # (B, H)
        global_feat = global_feat.unsqueeze(1).expand(-1, num_pos, -1)
        combined = torch.cat([local, global_feat], dim=2)  # (B, N, 2H)

        delta = self.decoder(combined).view(batch_size, num_pos, num_t_in, 3)
        last_frame = velocity_in[:, -1, :, :]
        out = last_frame.unsqueeze(2) + delta

        for i, idcs in enumerate(idcs_airfoil):
            out[i, idcs] = 0.0

        return out.transpose(1, 2)
