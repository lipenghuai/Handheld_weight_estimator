from pathlib import Path
import numpy as np
from plyfile import PlyData


def read_ply_xyz(ply_path: str | Path, mmap: bool = True) -> np.ndarray:
    """
    读取 PLY，返回 (N,3) float32 xyz。
    mmap=True 更快，但遇到某些异常文件可能更敏感；失败会在上层捕获。
    """
    ply_path = Path(ply_path)

    # 这里不在本函数吞异常：让上层决定“跳过还是报错”
    plydata = PlyData.read(str(ply_path), mmap=mmap)
    v = plydata["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz
