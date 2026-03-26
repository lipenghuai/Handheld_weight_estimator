import torch


def hausdorff_distance(x: torch.Tensor, y: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """
    近似 Hausdorff：
      H(x,y) = max( max_i min_j ||xi-yj||,  max_j min_i ||yj-xi|| )
    返回 scalar。
    """
    d = torch.cdist(x, y, p=2)
    if squared:
        d = d * d
    # x->y 每个点到 y 最近距离
    min_xy = d.min(dim=2).values  # (B, N)
    # y->x
    min_yx = d.min(dim=1).values  # (B, M)

    h_xy = min_xy.max(dim=1).values  # (B,)
    h_yx = min_yx.max(dim=1).values
    h = torch.max(h_xy, h_yx).mean()
    return h
