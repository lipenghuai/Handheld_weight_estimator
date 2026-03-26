import torch


def sinkhorn_emd(x: torch.Tensor, y: torch.Tensor, eps: float = 0.05, iters: int = 30) -> torch.Tensor:
    """
    近似 EMD（最优传输 OT）的 Sinkhorn 版本。
    注意：O(N^2) 计算，2048 点会偏慢；可在训练中低频率用或小 batch 用。
    x,y: (B, N, 3) 这里假设 N 相同（不相同也能跑，但意义略变）
    """
    B, N, _ = x.shape
    M = y.shape[1]

    # cost: (B, N, M)
    C = torch.cdist(x, y, p=2)  # (B,N,M)
    # Gibbs kernel
    K = torch.exp(-C / eps) + 1e-9

    # uniform marginals
    a = torch.full((B, N), 1.0 / N, device=x.device, dtype=x.dtype)
    b = torch.full((B, M), 1.0 / M, device=x.device, dtype=x.dtype)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn iterations
    for _ in range(iters):
        u = a / (K @ v.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
        v = b / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)

    # transport plan
    T = u.unsqueeze(-1) * K * v.unsqueeze(1)  # (B,N,M)
    emd = (T * C).sum(dim=(1, 2)).mean()
    return emd
