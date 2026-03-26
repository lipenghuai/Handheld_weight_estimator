import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.ply_io import read_ply_xyz
from utils.pc_sample import normalize_unit_sphere, random_sample, farthest_point_sample


class PLYAutoEncoderDataset(Dataset):
    """
    每条样本对应一个 ply 文件（以及可选的 rgb）。
    返回 dict：
      - points: (n_points, 3) float32 torch.Tensor
      - label : float (体重等；训练AE可忽略)
      - rgb   : PIL.Image 或 torch.Tensor (可选)
      - meta  : 路径等元信息
    """

    def __init__(
            self,
            index_jsonl: str | Path,
            root_dir: str | Path,
            n_points: int = 2048,
            sample_mode: str = "fps",  # "random" or "fps"
            fps_pre_n: int = 8192,
            normalize: bool = True,
            return_rgb: bool = True,
            rgb_to_tensor: bool = False,
            seed: int = 0,
            skip_bad: bool = True,
            bad_log_dir: str | Path | None = None,
    ):
        self.index_jsonl = Path(index_jsonl)
        self.root_dir = Path(root_dir)
        self.n_points = int(n_points)
        self.sample_mode = sample_mode
        self.fps_pre_n = int(fps_pre_n)
        self.normalize = bool(normalize)
        self.return_rgb = bool(return_rgb)
        self.rgb_to_tensor = bool(rgb_to_tensor)
        self.rng = np.random.default_rng(seed)

        self.items: List[Dict[str, Any]] = []
        with self.index_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        self.skip_bad = bool(skip_bad)
        self.bad_log_dir = Path(bad_log_dir) if bad_log_dir is not None and str(bad_log_dir) != "" else None
        if self.bad_log_dir is not None:
            self.bad_log_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.items)

    def _load_rgb(self, rgb_path: Path):
        img = Image.open(rgb_path).convert("RGB")
        if not self.rgb_to_tensor:
            return img
        # 不强制 torchvision，自己转 tensor：(3,H,W) float32 in [0,1]
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        # ply_path = self.root_dir / rec["ply_path"]
        # xyz = read_ply_xyz(ply_path)
        ply_path = self.root_dir / rec["ply_path"]
        try:
            xyz = read_ply_xyz(ply_path)
        except Exception as e:
            if not self.skip_bad:
                raise
            # 记录坏文件：每个 worker 写自己的文件，避免多进程抢同一个文件
            if self.bad_log_dir is not None:
                import os
                pid = os.getpid()
                log_path = self.bad_log_dir / f"bad_ply_{pid}.txt"
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(f"{ply_path}\t{type(e).__name__}\t{str(e)}\n")
            return None

        import zlib
        seed_i = zlib.crc32(str(ply_path).encode("utf-8")) & 0xffffffff
        self.rng = np.random.default_rng(seed_i)

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        if self.sample_mode == "random":
            xyz = random_sample(xyz, self.n_points, self.rng)
        elif self.sample_mode == "fps":
            xyz = farthest_point_sample(xyz, self.n_points, self.rng, pre_n=self.fps_pre_n)
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}, expected 'random' or 'fps'")

        points = torch.from_numpy(xyz.astype(np.float32))  # (n_points, 3)

        out: Dict[str, Any] = {
            "points": points,
            "label": float(rec.get("label", float("nan"))),
            "meta": {
                "folderA": rec.get("folderA"),
                "capture_root": rec.get("capture_root"),
                "ply_path": rec.get("ply_path"),
                "rgb_path": rec.get("rgb_path"),
                "rgb_reliable": rec.get("rgb_reliable"),
                "split": rec.get("split"),
            }
        }

        if self.return_rgb and rec.get("rgb_path") is not None:
            rgb_path = self.root_dir / rec["rgb_path"]
            if rgb_path.is_file():
                out["rgb"] = self._load_rgb(rgb_path)
            else:
                out["rgb"] = None
        else:
            out["rgb"] = None
        return out
