import torch
from torch.nn import Module


class LinearExtrapolation(Module):
    """Least-squares linear fit through the input frames, extrapolated to output times."""

    def forward(
        self,
        t: torch.Tensor,
        pos: torch.Tensor,
        idcs_airfoil: list[torch.Tensor],
        velocity_in: torch.Tensor,
    ) -> torch.Tensor:
        num_t_in = velocity_in.shape[1]
        t_in = t[:, :num_t_in, None, None]
        t_out = t[:, num_t_in:, None, None]

        mean_t = t_in.mean(dim=1, keepdim=True)
        mean_v = velocity_in.mean(dim=1, keepdim=True)

        dt = t_in - mean_t
        dv = velocity_in - mean_v

        slope = (dt * dv).sum(dim=1) / (dt * dt).sum(dim=1).clamp_min(1e-12)
        intercept = mean_v.squeeze(1) - slope * mean_t.squeeze(1)

        return intercept.unsqueeze(1) + slope.unsqueeze(1) * t_out
