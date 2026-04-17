import torch
from torch.nn import Module


class LastFrame(Module):
    """Persistence baseline: predict the last input frame for every output timestep."""

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        num_t_out = t.shape[1] - velocity_in.shape[1]
        last = velocity_in[:, -1:]
        return last.expand(-1, num_t_out, -1, -1).contiguous()
