import torch


def repulsion_loss(points: torch.Tensor, k: int = 8, h: float = 0.03) -> torch.Tensor:
    """
    Repulsion / Uniformity loss for point clouds.

    Args:
        points: (B, N, 3) torch.Tensor
        k: number of nearest neighbors (excluding itself)
        h: distance threshold. neighbors closer than h will be penalized.
           If your point cloud is normalized to unit sphere, h=0.02~0.05 is a good start.

    Returns:
        scalar torch.Tensor
    """
    assert points.dim() == 3 and points.size(-1) == 3, "points must be (B, N, 3)"
    B, N, _ = points.shape

    # Pairwise distances: (B, N, N)
    d = torch.cdist(points, points, p=2)

    # Exclude self-distance by adding a huge number on diagonal
    eye = torch.eye(N, device=points.device, dtype=points.dtype).unsqueeze(0)
    d = d + eye * 1e6

    # kNN distances: (B, N, k)  (smallest k distances)
    knn = d.topk(k=k, largest=False).values

    # Penalize neighbors that are too close: max(0, h - dist)^2
    penalty = torch.clamp(h - knn, min=0.0)
    loss = (penalty * penalty).mean()
    return loss
