import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_ch, hidden, out_ch, use_bn=True):
        super().__init__()
        layers = []
        c_in = in_ch
        for c in hidden:
            layers.append(nn.Linear(c_in, c))
            if use_bn:
                layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            c_in = c
        layers.append(nn.Linear(c_in, out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B*N, C)
        return self.net(x)


def build_2d_grid(n_points: int, device=None, dtype=None):
    """
    构造 2D folding 网格，shape: (n_points, 2)
    尽量接近正方形，不够则裁剪。
    """
    side = int(math.ceil(math.sqrt(n_points)))
    lin = torch.linspace(-1.0, 1.0, steps=side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (side^2, 2)
    grid = grid[:n_points]
    return grid


class FoldingDecoder(nn.Module):
    """
    两阶段 Folding Decoder：
      输入 latent (B, latent_dim)
      输出点云 (B, n_points, 3)
    可通过 width_mult 调整大小。
    """
    def __init__(
        self,
        latent_dim: int = 256,
        n_points: int = 2048,
        width_mult: float = 1.0,
        use_bn: bool = True,
    ):
        super().__init__()
        def c(x):
            return max(8, int(x * width_mult))

        self.latent_dim = latent_dim
        self.n_points = n_points

        # folding stage 1: [z, grid2] -> xyz
        self.fold1 = MLP(
            in_ch=latent_dim + 2,
            hidden=[c(512), c(512), c(256)],
            out_ch=3,
            use_bn=use_bn
        )

        # folding stage 2: [z, xyz1] -> refine xyz
        self.fold2 = MLP(
            in_ch=latent_dim + 3,
            hidden=[c(512), c(512), c(256)],
            out_ch=3,
            use_bn=use_bn
        )

        # 网格作为 buffer，方便一键导入/保存权重
        grid = build_2d_grid(n_points)
        self.register_buffer("grid2d", grid, persistent=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        return: (B, n_points, 3)
        """
        B = z.shape[0]
        device = z.device
        dtype = z.dtype

        grid = self.grid2d.to(device=device, dtype=dtype)               # (P, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1)                      # (B, P, 2)

        z_expand = z.unsqueeze(1).expand(-1, self.n_points, -1)         # (B, P, D)

        # stage 1
        x1 = torch.cat([z_expand, grid], dim=-1)                        # (B, P, D+2)
        x1 = x1.reshape(B * self.n_points, -1)
        xyz1 = self.fold1(x1).reshape(B, self.n_points, 3)

        # stage 2 refine
        x2 = torch.cat([z_expand, xyz1], dim=-1)                        # (B, P, D+3)
        x2 = x2.reshape(B * self.n_points, -1)
        delta = self.fold2(x2).reshape(B, self.n_points, 3)

        xyz = xyz1 + delta
        return xyz
