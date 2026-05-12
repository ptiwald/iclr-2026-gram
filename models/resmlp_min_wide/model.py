"""Rung C: same skeleton as ResMLPMin with the U-Net sidecar widened.

Trunk width (`hidden=256`) is unchanged. Only the internal channel base of the
voxel U-Net (`ch_base`) is bumped 64 -> 96, matching ResMLP's m2/m4 members
(~15M params). Tests whether the sidecar saturates at ch_base=64 or whether
extra 3D-conv capacity buys further lift over rung B.

`norm_stats.pt` is loaded by the parent class from `models/resmlp_min/`; the
wide variant shares the same per-channel velocity statistics.
"""
from __future__ import annotations

from models.resmlp_min.model import ResMLPMin


class ResMLPMinWide(ResMLPMin):

    def __init__(
        self,
        hidden: int = 256,
        num_pre: int = 2,
        num_post: int = 4,
        grid: tuple[int, int, int] = (64, 32, 32),
        ch_base: int = 96,
        num_pos_freqs: int = 10,
        num_vel_freqs: int = 3,
        num_dist_freqs: int = 6,
    ):
        super().__init__(
            hidden=hidden,
            num_pre=num_pre,
            num_post=num_post,
            grid=grid,
            ch_base=ch_base,
            num_pos_freqs=num_pos_freqs,
            num_vel_freqs=num_vel_freqs,
            num_dist_freqs=num_dist_freqs,
        )
