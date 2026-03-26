#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 index(json/jsonl) 中读取 ply_path，按 train_weight_regressor_flex1_r2.py 的逻辑：
1) 读入点云
2) normalize_unit_sphere_with_cr
3) random / fps 采样
4) 用 PointCloudAE.encoder 提取 z
5) 保存 z、cr 以及与 rgb 对应所需的元信息

输出文件为: {index_stem}_ply_latents.pt
其中包含：
- items: List[Dict]，每条记录里有 sample_id / ply_path / rgb_path / label 等
- z:    Tensor [N, latent_dim]
- cr:   Tensor [N, 4]
- feat_zcr: Tensor [N, latent_dim + 4]
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from plyfile import PlyData

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from models import PointCloudAE


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def norm_rel_path(p: str) -> str:
    p = str(p).replace("\\", "/")
    while "//" in p:
        p = p.replace("//", "/")
    return p


def resolve_rel_path(root_dir: Path, rel_path: str) -> Path:
    rel_norm = norm_rel_path(rel_path)
    if Path(rel_norm).is_absolute():
        return Path(rel_norm)
    return root_dir / Path(rel_norm)


def load_index_any(index_path: Path) -> List[Dict[str, Any]]:
    text = index_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # 先尝试整体 json
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if "items" in obj and isinstance(obj["items"], list):
                return obj["items"]
            return [obj]
    except Exception:
        pass

    # 回退 jsonl
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def build_sample_id(rec: Dict[str, Any]) -> str:
    key = "|".join([
        str(rec.get("split", "")),
        str(rec.get("folderA", "")),
        norm_rel_path(str(rec.get("folderA_path", ""))),
        norm_rel_path(str(rec.get("capture_root", ""))),
        norm_rel_path(str(rec.get("ply_path", ""))),
        norm_rel_path(str(rec.get("rgb_path", ""))),
        str(rec.get("label", "")),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def read_ply_xyz(ply_path: Path) -> np.ndarray:
    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz


def normalize_unit_sphere_with_cr(xyz: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float]:
    c = xyz.mean(axis=0, keepdims=True)
    xyz0 = xyz - c
    r = np.sqrt((xyz0 * xyz0).sum(axis=1)).max()
    r = float(max(r, eps))
    xyz_norm = (xyz0 / r).astype(np.float32)
    return xyz_norm, c.reshape(3).astype(np.float32), r


def random_sample(xyz: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    n = xyz.shape[0]
    if n >= n_points:
        idx = rng.choice(n, size=n_points, replace=False)
    else:
        idx = rng.choice(n, size=n_points, replace=True)
    return xyz[idx]


def farthest_point_sample(xyz: np.ndarray, n_points: int, rng: np.random.Generator, pre_n: int = 8192) -> np.ndarray:
    n = xyz.shape[0]
    if n > pre_n:
        xyz = random_sample(xyz, pre_n, rng)
        n = xyz.shape[0]

    centroids = np.zeros((n_points,), dtype=np.int64)
    dist = np.full((n,), 1e10, dtype=np.float32)

    farthest = int(rng.integers(0, n))
    for i in range(n_points):
        centroids[i] = farthest
        p = xyz[farthest:farthest + 1]
        d = ((xyz - p) ** 2).sum(axis=1).astype(np.float32)
        dist = np.minimum(dist, d)
        farthest = int(dist.argmax())

    return xyz[centroids]


class PLYIndexDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        root_dir: Path,
        n_points: int,
        sample_mode: str = "fps",
        fps_pre_n: int = 8192,
        seed: int = 0,
    ):
        self.items = items
        self.root_dir = Path(root_dir)
        self.n_points = int(n_points)
        self.sample_mode = sample_mode
        self.fps_pre_n = int(fps_pre_n)
        self.base_seed = int(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        ply_path = resolve_rel_path(self.root_dir, rec["ply_path"])

        xyz = read_ply_xyz(ply_path)
        xyz_norm, c, r = normalize_unit_sphere_with_cr(xyz)

        rng = np.random.default_rng(self.base_seed + idx)
        if self.sample_mode == "random":
            pts = random_sample(xyz_norm, self.n_points, rng)
        elif self.sample_mode == "fps":
            pts = farthest_point_sample(xyz_norm, self.n_points, rng, pre_n=self.fps_pre_n)
        else:
            raise ValueError("sample_mode must be 'random' or 'fps'")

        cr = np.concatenate([c, np.array([r], dtype=np.float32)], axis=0)

        meta = {
            "sample_id": build_sample_id(rec),
            "line_idx": idx,
            "split": rec.get("split", ""),
            "folderA": rec.get("folderA", ""),
            "folderA_path": rec.get("folderA_path", ""),
            "capture_root": rec.get("capture_root", ""),
            "ply_path": rec.get("ply_path", ""),
            "rgb_path": rec.get("rgb_path", ""),
            "label": float(rec.get("label", float("nan"))),
            "rgb_reliable": rec.get("rgb_reliable", None),
        }

        return {
            "points": torch.from_numpy(pts.astype(np.float32)),
            "cr": torch.from_numpy(cr.astype(np.float32)),
            "meta": meta,
        }


def collate_ply(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "points": torch.stack([b["points"] for b in batch], dim=0),  # [B, N, 3]
        "cr": torch.stack([b["cr"] for b in batch], dim=0),          # [B, 4]
        "meta": [b["meta"] for b in batch],
    }


@torch.no_grad()
def extract_one_index(
    index_path: Path,
    root_dir: Path,
    out_dir: Path,
    encoder: torch.nn.Module,
    device: torch.device,
    n_points: int,
    latent_dim: int,
    batch_size: int,
    num_workers: int,
    sample_mode: str,
    fps_pre_n: int,
    seed: int,
):
    items = load_index_any(index_path)
    if len(items) == 0:
        raise RuntimeError(f"index 为空: {index_path}")

    ds = PLYIndexDataset(
        items=items,
        root_dir=root_dir,
        n_points=n_points,
        sample_mode=sample_mode,
        fps_pre_n=fps_pre_n,
        seed=seed,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_ply,
    )

    all_z = []
    all_cr = []
    all_items = []

    pbar = tqdm(loader, desc=f"PLY encode | {index_path.name}", dynamic_ncols=True)
    for batch in pbar:
        pts = batch["points"].to(device, non_blocking=True)
        cr = batch["cr"].cpu()
        z = encoder(pts).detach().cpu()
        if z.ndim != 2:
            raise RuntimeError(f"PLY encoder 输出必须是 [B,D]，当前得到 {tuple(z.shape)}")
        if z.shape[1] != latent_dim:
            raise RuntimeError(f"latent_dim 不匹配: got {z.shape[1]}, expect {latent_dim}")

        all_z.append(z)
        all_cr.append(cr)
        all_items.extend(batch["meta"])

    z = torch.cat(all_z, dim=0)
    cr = torch.cat(all_cr, dim=0)
    feat_zcr = torch.cat([z, cr], dim=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{index_path.stem}_ply_latents.pt"
    payload = {
        "type": "ply_latents",
        "source_index": str(index_path),
        "root_dir": str(root_dir),
        "num_samples": len(all_items),
        "n_points": n_points,
        "latent_dim": latent_dim,
        "sample_mode": sample_mode,
        "fps_pre_n": fps_pre_n,
        "items": all_items,
        "z": z,
        "cr": cr,
        "feat_zcr": feat_zcr,
    }
    torch.save(payload, str(out_path))
    print(f"[OK] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="批量提取 PLY encoder latent")
    ap.add_argument("--root_dir", type=str, default="/2024219001/data/handheld_pigweight/data")
    ap.add_argument("--index_paths", type=str, nargs="+", required=True, help="可同时传 train/test 两个 index")
    ap.add_argument("--ae_ckpt", type=str, default=r"./runstrain/train256dgcnn4096/checkpoints/latest.pth")
    ap.add_argument("--ae_n_points", type=int, default=-1)
    ap.add_argument("--ae_latent_dim", type=int, default=-1)

    ap.add_argument("--sample_mode", type=str, default="fps", choices=["fps", "random"])
    ap.add_argument("--fps_pre_n", type=int, default=8192)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="precomputed_latents/ply")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    ae_args = ckpt.get("args", {})

    n_points = int(ae_args.get("n_points", -1))
    latent_dim = int(ae_args.get("latent_dim", -1))
    if n_points <= 0:
        n_points = args.ae_n_points
    if latent_dim <= 0:
        latent_dim = args.ae_latent_dim
    if n_points <= 0 or latent_dim <= 0:
        raise ValueError("无法确定 n_points/latent_dim，请检查 ckpt 或手动传参")

    ae = PointCloudAE(
        n_points=n_points,
        latent_dim=latent_dim,
        width_mult=float(ae_args.get("width_mult", 1.0)),
        enc_dropout=float(ae_args.get("enc_dropout", 0.0)),
        use_bn=bool(ae_args.get("use_bn", True)),
    )
    state = ckpt.get("model_state", None) or ckpt
    ae.load_state_dict(state, strict=True)
    ae = ae.to(device).eval()
    encoder = ae.encoder.eval()

    print(f"[INFO] device={device}")
    print(f"[INFO] n_points={n_points}, latent_dim={latent_dim}")

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    for index_path_str in args.index_paths:
        extract_one_index(
            index_path=Path(index_path_str),
            root_dir=root_dir,
            out_dir=out_dir,
            encoder=encoder,
            device=device,
            n_points=n_points,
            latent_dim=latent_dim,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_mode=args.sample_mode,
            fps_pre_n=args.fps_pre_n,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
