import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """对 (B, N, C) 做逐点 MLP：等价于 1x1 Conv。"""
    def __init__(self, in_ch, out_ch, bn=True, act=True):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch) if bn else None
        self.act = act

    def forward(self, x):
        # x: (B, N, C)
        x = self.lin(x)  # (B, N, out)
        if self.bn is not None:
            # BN1d 需要 (B, C, N)
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)
        if self.act:
            x = F.relu(x, inplace=True)
        return x


class PointNetEncoder(nn.Module):
    """
    PointNet 风格 encoder：
      输入:  (B, N, 3)
      输出:  latent (B, latent_dim)
    可通过 width_mult 调整模型大小。
    """
    def __init__(
        self,
        latent_dim: int = 256,
        width_mult: float = 1.0,
        dropout: float = 0.0,
        use_bn: bool = True,
    ):
        super().__init__()
        def c(x):  # scale channels
            return max(8, int(x * width_mult))

        self.mlp1 = SharedMLP(3,   c(64),  bn=use_bn, act=True)
        self.mlp2 = SharedMLP(c(64), c(128), bn=use_bn, act=True)
        self.mlp3 = SharedMLP(c(128), c(256), bn=use_bn, act=True)
        self.mlp4 = SharedMLP(c(256), c(512), bn=use_bn, act=True)

        self.fc1 = nn.Linear(c(512), c(512))
        self.bn1 = nn.LayerNorm(c(512)) if use_bn else None
        self.dp = nn.Dropout(dropout) if dropout > 0 else None
        self.fc2 = nn.Linear(c(512), latent_dim)

        # 让 latent 更稳定一点（可选）
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (B, N, 3)
        """
        x = self.mlp1(points)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)                     # (B, N, C)

        x = x.max(dim=1).values              # 全局 max pool -> (B, C)

        x = self.fc1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        if self.dp is not None:
            x = self.dp(x)

        z = self.fc2(x)                      # (B, latent_dim)
        z = self.latent_norm(z)
        return z
