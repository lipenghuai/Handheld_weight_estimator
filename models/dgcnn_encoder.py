import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, C)
    return idx: (B, N, k)  每个点的 kNN 索引（不含自身也可以含，影响不大）
    """
    # 用 (x^2 + y^2 - 2xy) 计算 pairwise 距离，避免 torch.cdist 的额外开销
    # dist: (B, N, N)
    B, N, C = x.shape
    xx = (x ** 2).sum(dim=2, keepdim=True)              # (B, N, 1)
    inner = torch.bmm(x, x.transpose(1, 2))             # (B, N, N)
    dist = xx + xx.transpose(1, 2) - 2.0 * inner        # (B, N, N)

    # 取最小的 k 个（距离越小越近）
    # topk 默认取最大，所以用 largest=False
    idx = dist.topk(k=k, dim=-1, largest=False, sorted=False).indices  # (B, N, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int, idx: torch.Tensor | None = None) -> torch.Tensor:
    """
    构造 EdgeConv 的图特征：
      feature = concat( x_j - x_i, x_i )
    x: (B, N, C)
    idx: (B, N, k)
    return: (B, 2C, N, k)
    """
    B, N, C = x.shape
    if idx is None:
        idx = knn(x, k=k)

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N     # (B,1,1)
    idx = idx + idx_base                                                # (B,N,k)
    idx = idx.reshape(-1)                                               # (B*N*k,)

    x_flat = x.reshape(B * N, C)                                        # (B*N,C)
    neighbors = x_flat[idx].view(B, N, k, C)                            # (B,N,k,C)

    x_i = x.view(B, N, 1, C).expand(-1, -1, k, -1)                      # (B,N,k,C)
    edge = torch.cat([neighbors - x_i, x_i], dim=3)                     # (B,N,k,2C)
    edge = edge.permute(0, 3, 1, 2).contiguous()                        # (B,2C,N,k)
    return edge


class EdgeConv(nn.Module):
    """
    EdgeConv block:
      input:  (B, C, N)   -> 先转为图特征 (B,2C,N,k)
      output: (B, C_out, N)  (对 k 维 max 聚合)
    """
    def __init__(self, in_channels: int, out_channels: int, k: int = 20, use_bn: bool = True):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, N)
        """
        B, C, N = x.shape
        x_t = x.transpose(1, 2).contiguous()           # (B,N,C)
        idx = knn(x_t, k=self.k)                       # (B,N,k)

        feat = get_graph_feature(x_t, k=self.k, idx=idx)  # (B,2C,N,k)
        out = self.conv(feat)
        if self.bn is not None:
            out = self.bn(out)
        out = F.relu(out, inplace=True)
        out = out.max(dim=3).values                    # max over k -> (B,C_out,N)
        return out


class DGCNNEncoder(nn.Module):
    """
    DGCNN/EdgeConv Encoder（升级版）：
      输入:  (B,N,3)
      输出:  (B,latent_dim)
    参数：
      k：邻域大小，常用 10~30
      width_mult：通道缩放，控制模型大小
    """
    def __init__(
        self,
        latent_dim: int = 256,
        k: int = 20,
        width_mult: float = 1.0,
        dropout: float = 0.0,
        use_bn: bool = True,
        global_pool: str = "max+avg",  # "max" or "avg" or "max+avg"
    ):
        super().__init__()
        self.k = int(k)
        self.global_pool = global_pool

        def c(x):  # scale channels
            return max(16, int(x * width_mult))

        # EdgeConv layers（经典配置：64,64,128,256）
        self.ec1 = EdgeConv(in_channels=3, out_channels=c(64),  k=self.k, use_bn=use_bn)
        self.ec2 = EdgeConv(in_channels=c(64), out_channels=c(64),  k=self.k, use_bn=use_bn)
        self.ec3 = EdgeConv(in_channels=c(64), out_channels=c(128), k=self.k, use_bn=use_bn)
        self.ec4 = EdgeConv(in_channels=c(128), out_channels=c(256), k=self.k, use_bn=use_bn)

        feat_dim = c(64) + c(64) + c(128) + c(256)

        # 全局池化后再 MLP 到 latent
        if global_pool == "max+avg":
            fc_in = feat_dim * 2
        else:
            fc_in = feat_dim

        self.fc1 = nn.Linear(fc_in, c(512))
        self.bn1 = nn.LayerNorm(c(512)) if use_bn else None
        self.dp = nn.Dropout(dropout) if dropout > 0 else None
        self.fc2 = nn.Linear(c(512), latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (B,N,3)
        """
        # 转为 (B,3,N)
        x = points.transpose(1, 2).contiguous()

        x1 = self.ec1(x)   # (B,c64,N)
        x2 = self.ec2(x1)  # (B,c64,N)
        x3 = self.ec3(x2)  # (B,c128,N)
        x4 = self.ec4(x3)  # (B,c256,N)

        feat = torch.cat([x1, x2, x3, x4], dim=1)  # (B,feat_dim,N)

        if self.global_pool == "max":
            g = feat.max(dim=2).values
        elif self.global_pool == "avg":
            g = feat.mean(dim=2)
        elif self.global_pool == "max+avg":
            g_max = feat.max(dim=2).values
            g_avg = feat.mean(dim=2)
            g = torch.cat([g_max, g_avg], dim=1)
        else:
            raise ValueError(f"Unknown global_pool: {self.global_pool}")

        g = self.fc1(g)
        if self.bn1 is not None:
            g = self.bn1(g)
        g = F.relu(g, inplace=True)
        if self.dp is not None:
            g = self.dp(g)

        z = self.fc2(g)
        z = self.latent_norm(z)
        return z
