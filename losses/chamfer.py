import torch


def chamfer_distance_l2(x: torch.Tensor, y: torch.Tensor, squared: bool = True) -> torch.Tensor:
    """
    x: (B, N, 3)
    y: (B, M, 3)
    return: scalar loss (平均)
    """
    # (B, N, M)
    d = torch.cdist(x, y, p=2)
    if squared:
        d = d * d
    # x->y 最近点
    min_xy = d.min(dim=2).values  # (B, N)
    # y->x 最近点
    min_yx = d.min(dim=1).values  # (B, M)
    return (min_xy.mean() + min_yx.mean()) * 0.5


def chamfer_distance(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    x: (B, N, 3)
    y: (B, M, 3)
    p=1 or 2
    return: scalar
    """
    d = torch.cdist(x, y, p=2)  # 欧氏距离
    if p == 2:
        d = d * d
    elif p == 1:
        pass
    else:
        raise ValueError("p must be 1 or 2")

    min_xy = d.min(dim=2).values  # (B, N)
    min_yx = d.min(dim=1).values  # (B, M)
    return 0.5 * (min_xy.mean() + min_yx.mean())
# def chamfer_distance(
#         x: torch.Tensor,
#         y: torch.Tensor,
#         p: int = 2,
#         axis_weight: torch.Tensor | None = None,
# ) -> torch.Tensor:
#     """
#     x: (B, N, 3)
#     y: (B, M, 3)
#     p=1 or 2
#     axis_weight: (3,) 或 (1,1,3)；例如 [1,1,5] 表示 z 放大 5 倍（等价于加权距离）
#     return: scalar
#     """
#     if axis_weight is not None:
#         # 保证可广播到 (B,N,3)
#         if axis_weight.ndim == 1:
#             axis_weight = axis_weight.view(1, 1, 3)
#         axis_weight = axis_weight.to(device=x.device, dtype=x.dtype)
#         x = x * axis_weight
#         y = y * axis_weight
#
#     d = torch.cdist(x, y, p=2)  # 欧氏距离
#     if p == 2:
#         d = d * d
#     elif p == 1:
#         pass
#     else:
#         raise ValueError("p must be 1 or 2")
#
#     min_xy = d.min(dim=2).values  # (B, N)
#     min_yx = d.min(dim=1).values  # (B, M)
#     return 0.5 * (min_xy.mean() + min_yx.mean())


def chamfer_distance_split(x: torch.Tensor, y: torch.Tensor, p: int = 2):
    """
    返回两个方向的项，方便加权：
      L = w1 * x->y + w2 * y->x
    """
    d = torch.cdist(x, y, p=2)
    if p == 2:
        d = d * d
    min_xy = d.min(dim=2).values.mean()  # scalar
    min_yx = d.min(dim=1).values.mean()
    return min_xy, min_yx
