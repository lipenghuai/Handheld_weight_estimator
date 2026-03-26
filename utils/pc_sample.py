import numpy as np


def normalize_unit_sphere(xyz: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    中心化 + 缩放到单位球内（最大半径=1）。
    """
    c = xyz.mean(axis=0, keepdims=True)
    xyz0 = xyz - c
    r = np.sqrt((xyz0 * xyz0).sum(axis=1)).max()
    r = max(r, eps)
    return xyz0 / r


def random_sample(xyz: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    n = xyz.shape[0]
    if n >= n_points:
        idx = rng.choice(n, size=n_points, replace=False)
    else:
        idx = rng.choice(n, size=n_points, replace=True)
    return xyz[idx]


def farthest_point_sample(xyz: np.ndarray, n_points: int, rng: np.random.Generator, pre_n: int = 8192) -> np.ndarray:
    """
    简单 CPU-FPS（O(pre_n * n_points)），为避免太慢：
      - 若原始点很多，先随机下采样到 pre_n
      - 再做 FPS 到 n_points

    适合几万~几十万点的场景：先 random->8192，再 FPS->2048。
    """
    n = xyz.shape[0]
    if n > pre_n:
        xyz = random_sample(xyz, pre_n, rng)
        n = xyz.shape[0]

    # FPS
    centroids = np.zeros((n_points,), dtype=np.int64)
    dist = np.full((n,), 1e10, dtype=np.float32)

    farthest = int(rng.integers(0, n))
    for i in range(n_points):
        centroids[i] = farthest
        p = xyz[farthest:farthest + 1]                  # (1,3)
        d = ((xyz - p) ** 2).sum(axis=1).astype(np.float32)
        dist = np.minimum(dist, d)
        farthest = int(dist.argmax())

    return xyz[centroids]
