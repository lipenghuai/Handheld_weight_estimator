#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 index(json/jsonl) 中读取 rgb_path，按 infer_single_image.py 的预处理逻辑：
1) ResizeLongestSideAndPad
2) 输入 RGB autoencoder
3) 取 latent
4) 保存 latent 和与 PLY 对齐所需的元信息

输出文件为: {index_stem}_rgb_latents.pt
其中包含：
- items: List[Dict]
- latent: Tensor [N, C, H, W]
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from models import vgg, resnet


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


def build_model(arch: str) -> torch.nn.Module:
    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(arch)
        model = vgg.VGGAutoEncoder(configs)
    elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    else:
        raise ValueError(f"不支持的 arch: {arch}")
    return model


def smart_load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[nk] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f"[INFO] 已加载权重: {ckpt_path}")
    if missing:
        print(f"[WARN] missing keys 数量: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys 数量: {len(unexpected)}")


class ResizeLongestSideAndPad:
    def __init__(self, target_size=224, fill=0):
        self.target_size = target_size
        self.fill = fill

    @staticmethod
    def _get_resample_mode():
        if hasattr(Image, "Resampling"):
            return Image.Resampling.BILINEAR
        return Image.BILINEAR

    def get_params(self, img: Image.Image) -> Dict[str, int]:
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        return {
            "orig_w": w,
            "orig_h": h,
            "new_w": new_w,
            "new_h": new_h,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "target_size": self.target_size,
        }

    def resize_only(self, img: Image.Image):
        params = self.get_params(img)
        img_resized = img.resize((params["new_w"], params["new_h"]), resample=self._get_resample_mode())
        return img_resized, params

    def pad_resized(self, img_resized: Image.Image, params: Dict[str, int]) -> Image.Image:
        pad_x = params["pad_x"]
        pad_y = params["pad_y"]

        if img_resized.mode == "RGB":
            canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        elif img_resized.mode == "L":
            canvas = Image.new("L", (self.target_size, self.target_size), 0)
        else:
            canvas = Image.new(img_resized.mode, (self.target_size, self.target_size), self.fill)

        canvas.paste(img_resized, (pad_x, pad_y))
        return canvas

    def __call__(self, img: Image.Image):
        img_resized, params = self.resize_only(img)
        img_padded = self.pad_resized(img_resized, params)
        return img_padded, params


class RGBIndexDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], root_dir: Path, target_size: int = 224):
        self.items = items
        self.root_dir = Path(root_dir)
        self.transformer = ResizeLongestSideAndPad(target_size=target_size, fill=0)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        rgb_path = resolve_rel_path(self.root_dir, rec["rgb_path"])
        img = Image.open(rgb_path).convert("RGB")
        img_padded, resize_meta = self.transformer(img)
        inp = self.to_tensor(img_padded)

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
            "orig_w": resize_meta["orig_w"],
            "orig_h": resize_meta["orig_h"],
            "new_w": resize_meta["new_w"],
            "new_h": resize_meta["new_h"],
            "pad_x": resize_meta["pad_x"],
            "pad_y": resize_meta["pad_y"],
            "target_size": resize_meta["target_size"],
        }
        return {"image": inp, "meta": meta}


def collate_rgb(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "meta": [b["meta"] for b in batch],
    }


@torch.no_grad()
def forward_model(model: torch.nn.Module, inp: torch.Tensor):
    out, latent = model(inp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out, latent


@torch.no_grad()
def extract_one_index(
    index_path: Path,
    root_dir: Path,
    out_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    target_size: int,
):
    items = load_index_any(index_path)
    if len(items) == 0:
        raise RuntimeError(f"index 为空: {index_path}")

    ds = RGBIndexDataset(items=items, root_dir=root_dir, target_size=target_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_rgb,
    )

    all_latent = []
    all_items = []

    pbar = tqdm(loader, desc=f"RGB encode | {index_path.name}", dynamic_ncols=True)
    for batch in pbar:
        inp = batch["image"].to(device, non_blocking=True)
        _, latent = forward_model(model, inp)
        latent = latent.detach().cpu()
        if latent.ndim != 4:
            raise RuntimeError(f"RGB latent 预期为 [B,C,H,W]，当前得到 {tuple(latent.shape)}")
        all_latent.append(latent)
        all_items.extend(batch["meta"])

    latent = torch.cat(all_latent, dim=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{index_path.stem}_rgb_latents.pt"
    payload = {
        "type": "rgb_latents",
        "source_index": str(index_path),
        "root_dir": str(root_dir),
        "num_samples": len(all_items),
        "items": all_items,
        "latent": latent,
        "latent_shape": list(latent.shape[1:]),
        "target_size": target_size,
    }
    torch.save(payload, str(out_path))
    print(f"[OK] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="批量提取 RGB latent")
    parser.add_argument("--root_dir", type=str, default="/2024219001/data/handheld_pigweight/data")
    parser.add_argument("--index_paths", type=str, nargs="+", required=True, help="可同时传 train/test 两个 index")
    parser.add_argument("--arch", type=str, default="vgg16",
                        help="vgg11/vgg13/vgg16/vgg19/resnet18/34/50/101/152")
    parser.add_argument("--resume", type=str, default="./imagenet-vgg16.pth", help="RGB autoencoder 权重路径")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="/2024219001/lip/pointNET-AE-LIP/pointNET-AE-LIP/precomputed_latents/rgb")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[INFO] device = {device}")

    model = build_model(args.arch)
    model.to(device)
    model.eval()
    smart_load_checkpoint(model, args.resume, device)

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    for index_path_str in args.index_paths:
        extract_one_index(
            index_path=Path(index_path_str),
            root_dir=root_dir,
            out_dir=out_dir,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_size=args.target_size,
        )


if __name__ == "__main__":
    main()
