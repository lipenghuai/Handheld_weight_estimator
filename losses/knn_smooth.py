import torch


def knn_edge_length_loss(x: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    让点云的局部边长不要太极端（避免局部爆炸或塌缩）
    这里简单用：邻居距离的方差/均值 作为正则。
    """
    B, N, _ = x.shape
    d = torch.cdist(x, x, p=2)
    d = d + torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0) * 1e6
    knn = d.topk(k=k, largest=False).values  # (B, N, k)

    mu = knn.mean(dim=2)           # (B, N)
    var = knn.var(dim=2)           # (B, N)
    # 归一化，避免尺度影响太大
    loss = (var / (mu * mu + 1e-6)).mean()
    return loss
