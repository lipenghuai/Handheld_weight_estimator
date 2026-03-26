import os
import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 让 “python train_weight_regressor_flex.py” 能直接 import 本项目
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from plyfile import PlyData
from models import PointCloudAE


# -------------------------
# 基础工具
# -------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_ply_xyz(ply_path: Path) -> np.ndarray:
    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    return xyz


def normalize_unit_sphere_with_cr(xyz: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    你给的归一化：
      c = mean(xyz)
      r = max(||xyz-c||)
      xyz_norm = (xyz - c) / r
    返回 xyz_norm, c(3,), r(scalar)
    """
    c = xyz.mean(axis=0, keepdims=True)  # (1,3)
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
    """
    CPU-FPS：先随机到 pre_n，再 FPS 到 n_points。
    """
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


def load_index_jsonl(index_path: Path) -> List[Dict[str, Any]]:
    items = []
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def set_batchnorm_eval(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()


# -------------------------
# Dataset（points + cr + label）
# -------------------------
class PLYForZDataset(Dataset):
    def __init__(
        self,
        index_items: List[Dict[str, Any]],
        root_dir: Path,
        n_points: int,
        sample_mode: str = "fps",
        fps_pre_n: int = 8192,
        seed: int = 0,
        max_per_folderA: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.n_points = int(n_points)
        self.sample_mode = sample_mode
        self.fps_pre_n = int(fps_pre_n)
        self.rng = np.random.default_rng(seed)

        items = index_items

        # 可选：限制每个 folderA 最大帧数，避免某些猪帧数过多导致训练偏置
        if max_per_folderA and max_per_folderA > 0:
            byA: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                a = str(it.get("folderA", ""))
                byA.setdefault(a, []).append(it)
            new_items = []
            for a, lst in byA.items():
                if len(lst) <= max_per_folderA:
                    new_items.extend(lst)
                else:
                    idx = self.rng.choice(len(lst), size=max_per_folderA, replace=False)
                    for j in idx:
                        new_items.append(lst[int(j)])
            items = new_items

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        ply_rel = rec["ply_path"]
        label = float(rec.get("label", float("nan")))
        ply_path = self.root_dir / ply_rel

        xyz = read_ply_xyz(ply_path)
        xyz_norm, c, r = normalize_unit_sphere_with_cr(xyz)

        if self.sample_mode == "random":
            pts = random_sample(xyz_norm, self.n_points, self.rng)
        elif self.sample_mode == "fps":
            pts = farthest_point_sample(xyz_norm, self.n_points, self.rng, pre_n=self.fps_pre_n)
        else:
            raise ValueError("sample_mode must be 'random' or 'fps'")

        cr = np.concatenate([c, np.array([r], dtype=np.float32)], axis=0)  # (4,)

        return {
            "points": torch.from_numpy(pts.astype(np.float32)),
            "cr": torch.from_numpy(cr.astype(np.float32)),
            "label": label,
            "meta": {
                "folderA": rec.get("folderA", ""),
                "ply_path": rec.get("ply_path", ""),
            }
        }


class RetryDataset(Dataset):
    """
    若样本读取失败/无效，则随机换 index 重试，避免 loader 因单个坏文件崩掉。
    """
    def __init__(self, base: Dataset, max_retry: int = 10, seed: int = 0):
        self.base = base
        self.max_retry = int(max_retry)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        n = len(self.base)
        cur = idx
        for _ in range(self.max_retry):
            try:
                s = self.base[cur]
                pts = s.get("points", None)
                cr = s.get("cr", None)
                y = s.get("label", float("nan"))
                if isinstance(pts, torch.Tensor) and pts.ndim == 2 and pts.shape[1] == 3 and pts.numel() > 0 \
                   and isinstance(cr, torch.Tensor) and cr.numel() == 4 \
                   and (not math.isnan(float(y))):
                    return s
            except Exception:
                pass
            cur = int(self.rng.integers(0, n))
        return {"points": None, "cr": None, "label": float("nan"), "meta": {"bad": True, "orig_idx": idx}}


def collate_z(batch: List[Any]) -> Optional[Dict[str, Any]]:
    valid = []
    for b in batch:
        if not isinstance(b, dict):
            continue
        pts = b.get("points", None)
        cr = b.get("cr", None)
        y = b.get("label", float("nan"))
        if isinstance(pts, torch.Tensor) and pts.ndim == 2 and pts.shape[1] == 3 and pts.numel() > 0 \
           and isinstance(cr, torch.Tensor) and cr.numel() == 4 \
           and (not math.isnan(float(y))):
            valid.append(b)
    if len(valid) == 0:
        return None
    return {
        "points": torch.stack([b["points"] for b in valid], dim=0),  # (B,N,3)
        "cr": torch.stack([b["cr"] for b in valid], dim=0),          # (B,4)
        "label": torch.tensor([b["label"] for b in valid], dtype=torch.float32),  # (B,)
        "meta": [b.get("meta", {}) for b in valid],
    }


# -------------------------
# 回归器：Torch MLP（可反传）
# -------------------------
class TorchMLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: str = "256,128", dropout: float = 0.1):
        super().__init__()
        hs = [int(x) for x in hidden.split(",") if x.strip()]
        layers: List[nn.Module] = []
        d = in_dim
        for h in hs:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -------------------------
# sklearn 回归器工厂（冻结 encoder 专用）
# -------------------------
def _require_sklearn():
    try:
        import sklearn  # noqa
        import joblib   # noqa
    except Exception as e:
        raise RuntimeError("需要安装 scikit-learn 与 joblib：pip install scikit-learn joblib") from e


def make_sklearn_model(name: str, params: Dict[str, Any], seed: int):
    _require_sklearn()
    name = name.lower()

    from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    if name == "ridge":
        return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=seed)
    if name == "lasso":
        return Lasso(alpha=float(params.get("alpha", 1e-3)), max_iter=int(params.get("max_iter", 5000)), random_state=seed)
    if name == "elasticnet":
        return ElasticNet(alpha=float(params.get("alpha", 1e-3)), l1_ratio=float(params.get("l1_ratio", 0.5)),
                          max_iter=int(params.get("max_iter", 5000)), random_state=seed)
    if name == "svr":
        return SVR(C=float(params.get("C", 10.0)), epsilon=float(params.get("epsilon", 0.1)), kernel=str(params.get("kernel", "rbf")))
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=int(params.get("k", 8)), weights=str(params.get("weights", "distance")))
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=seed,
            n_jobs=int(params.get("n_jobs", -1)),
        )
    if name == "gbrt":
        return GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("lr", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=seed,
        )
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 600)),
            max_depth=params.get("max_depth", None),
            random_state=seed,
            n_jobs=int(params.get("n_jobs", -1)),
        )
    if name == "adaboost":
        base = DecisionTreeRegressor(max_depth=int(params.get("max_depth", 3)), random_state=seed)
        # 兼容新旧 sklearn
        try:
            return AdaBoostRegressor(
                estimator=base,
                n_estimators=int(params.get("n_estimators", 500)),
                learning_rate=float(params.get("lr", 0.05)),
                loss=str(params.get("loss", "linear")),
                random_state=seed,
            )
        except TypeError:
            return AdaBoostRegressor(
                base_estimator=base,
                n_estimators=int(params.get("n_estimators", 500)),
                learning_rate=float(params.get("lr", 0.05)),
                loss=str(params.get("loss", "linear")),
                random_state=seed,
            )
    if name == "sgd":
        # 支持 partial_fit
        return SGDRegressor(
            loss=str(params.get("loss", "huber")),
            alpha=float(params.get("alpha", 1e-4)),
            learning_rate=str(params.get("learning_rate", "invscaling")),
            eta0=float(params.get("eta0", 1e-3)),
            random_state=seed,
        )

    raise ValueError(f"Unknown sklearn reg_model: {name}")


def sklearn_save(obj, path: Path):
    _require_sklearn()
    import joblib
    joblib.dump(obj, str(path))


# -------------------------
# 指标
# -------------------------
def mae_t(pred, y): 
    return (pred - y).abs().mean()

def rmse_t(pred, y): 
    return torch.sqrt(((pred - y) ** 2).mean() + 1e-12)

def mape_t(pred, y): 
    return ((pred - y).abs() / (y.abs() + 1e-6)).mean() * 100.0

def r2_t(pred, y):
    # R^2 = 1 - SS_res/SS_tot；在 y 方差极小/为0时做数值保护
    y_mean = y.mean()
    ss_tot = ((y - y_mean) ** 2).sum()
    ss_res = ((y - pred) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def r2_np(pred: np.ndarray, y: np.ndarray) -> float:
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# -------------------------
# 日志
# -------------------------
class CSVLogger:
    def __init__(self, csv_path: Path, header: List[str]):
        self.csv_path = csv_path
        self.header = header
        self._f = self.csv_path.open("w", encoding="utf-8", newline="")
        self._f.write(",".join(header) + "\n")
        self._f.flush()

    def log(self, row: Dict[str, Any]):
        vals = []
        for k in self.header:
            v = row.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.8f}")
            else:
                vals.append(str(v))
        self._f.write(",".join(vals) + "\n")

    def flush(self): self._f.flush()
    def close(self): self._f.flush(); self._f.close()


# -------------------------
# 特征提取与统计
# -------------------------
@torch.no_grad()
def extract_features_zcr_to_cpu(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    encoder.eval()
    feats, labels, paths = [], [], []
    for batch in tqdm(loader, desc="Extract z+cr", dynamic_ncols=True):
        if batch is None:
            continue
        pts = batch["points"].to(device, non_blocking=True)
        cr = batch["cr"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        z = encoder(pts)
        if z.shape[1] != latent_dim:
            raise RuntimeError(f"latent_dim mismatch: got {z.shape[1]} expected {latent_dim}")
        feat = torch.cat([z, cr], dim=1)
        feats.append(feat.detach().cpu())
        labels.append(y.detach().cpu())
        metas = batch.get("meta", [])
        for meta in metas:
            if isinstance(meta, dict):
                paths.append(str(meta.get("ply_path", "")))
            else:
                paths.append("")
    X = torch.cat(feats, dim=0) if feats else torch.empty(0, latent_dim + 4)
    y = torch.cat(labels, dim=0) if labels else torch.empty(0)
    return X, y, paths


def save_prediction_table(save_path: Path, sample_paths: List[str], labels: Any, preds: Any) -> pd.DataFrame:
    labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
    preds_np = np.asarray(preds, dtype=np.float32).reshape(-1)
    n = min(len(sample_paths), len(labels_np), len(preds_np))
    df = pd.DataFrame({
        "sample_path": list(sample_paths)[:n],
        "label": labels_np[:n],
        "pred": preds_np[:n],
    })
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


def save_best_info(save_path: Path, info: Dict[str, Any]):
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def collect_predictions_finetune(
    encoder: nn.Module,
    reg: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    feat_zscore: bool,
    denorm_y_fn,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    encoder.eval()
    reg.eval()
    sample_paths: List[str] = []
    labels_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    feat_mean_dev = feat_mean.to(device)
    feat_std_dev = feat_std.to(device)

    for batch in loader:
        if batch is None:
            continue
        pts = batch["points"].to(device, non_blocking=True)
        cr = batch["cr"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        z = encoder(pts)
        feat = torch.cat([z, cr], dim=1)
        if feat_zscore:
            feat = (feat - feat_mean_dev) / feat_std_dev

        pred_n = reg(feat)
        pred_kg = denorm_y_fn(pred_n).detach().cpu().numpy().astype(np.float32)
        y_kg = y.detach().cpu().numpy().astype(np.float32)

        preds_all.append(pred_kg)
        labels_all.append(y_kg)

        metas = batch.get("meta", [])
        for meta in metas:
            if isinstance(meta, dict):
                sample_paths.append(str(meta.get("ply_path", "")))
            else:
                sample_paths.append("")

    labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,), dtype=np.float32)
    preds_np = np.concatenate(preds_all, axis=0) if preds_all else np.empty((0,), dtype=np.float32)
    return sample_paths, labels_np, preds_np


@torch.no_grad()
def estimate_feat_stats_welford(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    latent_dim: int,
    max_samples: int = 5000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    微调 encoder 模式下：在训练前用当前 encoder 估计一次 [z,c,r] 的 mean/std（固定使用）。
    为了省时默认最多取 max_samples 条样本。
    """
    encoder.eval()
    count = 0
    mean = None
    m2 = None

    for batch in tqdm(loader, desc="Estimate feat mean/std", dynamic_ncols=True):
        if batch is None:
            continue
        pts = batch["points"].to(device, non_blocking=True)
        cr = batch["cr"].to(device, non_blocking=True)
        z = encoder(pts)
        feat = torch.cat([z, cr], dim=1)  # (B, D+4)
        feat_cpu = feat.detach().cpu()

        B = feat_cpu.shape[0]
        for i in range(B):
            x = feat_cpu[i]
            if mean is None:
                mean = x.clone()
                m2 = torch.zeros_like(x)
                count = 1
            else:
                count += 1
                delta = x - mean
                mean = mean + delta / count
                delta2 = x - mean
                m2 = m2 + delta * delta2
            if count >= max_samples:
                break
        if count >= max_samples:
            break

    if mean is None:
        raise RuntimeError("无法估计特征统计：可能训练数据全被过滤为 None。")

    var = m2 / max(1, count - 1)
    std = torch.sqrt(var.clamp_min(1e-12))
    return mean.unsqueeze(0), std.unsqueeze(0)


# -------------------------
# 主流程
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root_dir", type=str, default="/2024219001/data/handheld_pigweight/data")
    ap.add_argument("--train_index", type=str, default="indices/train.jsonl")
    ap.add_argument("--test_index", type=str, default="indices/test.jsonl")

    ap.add_argument("--ae_ckpt", type=str, default=r"./runstrain/train256dgcnn4096/checkpoints/latest.pth")
    ap.add_argument("--ae_n_points", type=int, default=-1)
    ap.add_argument("--ae_latent_dim", type=int, default=-1)

    ap.add_argument("--sample_mode", type=str, default="fps", choices=["fps", "random"])
    ap.add_argument("--fps_pre_n", type=int, default=8192)
    ap.add_argument("--max_per_folderA", type=int, default=0)
    ap.add_argument("--max_retry", type=int, default=10)

    ap.add_argument("--out_root", type=str, default="runs_weight")
    ap.add_argument("--run_name", type=str, default="train256dgcnn4096")

    # 回归器选择
    ap.add_argument("--reg_model", type=str, default="mlp",
                    choices=["mlp", "ridge", "lasso", "elasticnet", "svr", "knn", "rf", "gbrt", "extratrees", "adaboost", "sgd"])
    ap.add_argument("--reg_params", type=str, default="{}", help="sklearn 参数json，如 '{\"alpha\":2.0}'")
    ap.add_argument("--hidden", type=str, default="128")  # 仅 mlp
    ap.add_argument("--dropout", type=float, default=0.2)     # 仅 mlp

    ap.add_argument("--epochs", type=int, default=200)        # mlp/sgd 有意义
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--enc_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=10)

    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)

    # 标准化开关（强烈建议保持默认）
    ap.add_argument("--label_zscore", type=int, default=1, help="是否对体重做z-score训练（建议1）")
    ap.add_argument("--feat_zscore", type=int, default=1, help="是否对输入特征[z,c,r]做z-score（建议1）")
    ap.add_argument("--feat_stats_max_samples", type=int, default=500000, help="finetune模式估计feat均值方差的最大样本数")

    ap.add_argument("--loss", type=str, default="huber", choices=["huber", "mse", "mae"])

    # 微调 encoder（仅 mlp 支持）
    ap.add_argument("--finetune_encoder", type=int, default=0, choices=[0, 1])
    ap.add_argument("--encoder_bn_eval", type=int, default=1, choices=[0, 1], help="batch=1 微调时建议1")

    # 冻结模式特征缓存
    ap.add_argument("--rebuild_features", type=int, default=0, choices=[0, 1])

    # debug
    ap.add_argument("--debug_stats", type=int, default=1, choices=[0, 1])

    args = ap.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    reg_params = json.loads(args.reg_params) if args.reg_params.strip() else {}

    if args.reg_model != "mlp" and args.finetune_encoder:
        raise ValueError("sklearn 回归器不支持 finetune_encoder=1。请用 --reg_model mlp 或把 finetune_encoder 设为 0。")

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / (args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    log_dir = run_dir / "logs"; log_dir.mkdir(exist_ok=True)
    feat_dir = run_dir / "features"; feat_dir.mkdir(exist_ok=True)
    pred_dir = run_dir / "best_predictions"; pred_dir.mkdir(exist_ok=True)
    best_info_path = run_dir / "best_model_info.json"

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Run dir: {run_dir}")
    print(f"Device : {device}")
    print(f"reg_model={args.reg_model}, finetune_encoder={args.finetune_encoder}, feat_zscore={args.feat_zscore}, label_zscore={args.label_zscore}")

    best_test_mae = float("inf")
    best_epoch = -1
    best_model_path = None

    # -------------------------
    # Load AE（只用 encoder）
    # -------------------------
    ckpt = torch.load(args.ae_ckpt, map_location="cpu")
    ae_args = ckpt.get("args", {})

    n_points = int(ae_args.get("n_points", -1))
    latent_dim = int(ae_args.get("latent_dim", -1))
    if n_points <= 0: n_points = args.ae_n_points
    if latent_dim <= 0: latent_dim = args.ae_latent_dim
    if n_points <= 0 or latent_dim <= 0:
        raise ValueError("无法确定 n_points/latent_dim：请确保 AE ckpt 里有 args 或传 --ae_n_points/--ae_latent_dim")

    ae = PointCloudAE(
        n_points=n_points,
        latent_dim=latent_dim,
        width_mult=float(ae_args.get("width_mult", 1.0)),
        enc_dropout=float(ae_args.get("enc_dropout", 0.0)),
        use_bn=bool(ae_args.get("use_bn", True)),
    )
    state = ckpt.get("model_state", None) or ckpt
    ae.load_state_dict(state, strict=True)
    ae = ae.to(device)

    encoder = ae.encoder
    ae.decoder.eval()
    for p in ae.decoder.parameters():
        p.requires_grad_(False)

    in_dim = latent_dim + 4

    # -------------------------
    # Dataset
    # -------------------------
    root_dir = Path(args.root_dir)
    train_items = load_index_jsonl(Path(args.train_index))
    test_items = load_index_jsonl(Path(args.test_index))

    # label 统计（kg）
    y_train_all = []
    for it in train_items:
        try:
            y = float(it.get("label", float("nan")))
            if not math.isnan(y):
                y_train_all.append(y)
        except Exception:
            pass
    if len(y_train_all) == 0:
        raise RuntimeError("训练集 label 为空/全 NaN：请检查 index.jsonl 的 label 字段")

    y_mean = float(np.mean(y_train_all))
    y_std = float(np.std(y_train_all) + 1e-6)

    def norm_y(y: torch.Tensor) -> torch.Tensor:
        return (y - y_mean) / y_std if args.label_zscore else y

    def denorm_y(y: torch.Tensor) -> torch.Tensor:
        return y * y_std + y_mean if args.label_zscore else y

    base_train = PLYForZDataset(train_items, root_dir, n_points, args.sample_mode, args.fps_pre_n, args.seed, args.max_per_folderA)
    base_test  = PLYForZDataset(test_items,  root_dir, n_points, args.sample_mode, args.fps_pre_n, args.seed + 123, args.max_per_folderA)
    train_set = RetryDataset(base_train, args.max_retry, args.seed)
    test_set  = RetryDataset(base_test,  args.max_retry, args.seed + 999)

    loader_train = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_z
    )
    loader_test = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_z
    )

    # logs（loss 是“训练用损失”（label_zscore空间），不是 NaN）
    header = ["time", "split", "epoch", "iter", "loss", "mae_kg", "rmse_kg", "mape", "r2", "lr_reg", "lr_enc"]
    train_logger = CSVLogger(log_dir / "train_iter_log.csv", header)
    test_logger  = CSVLogger(log_dir / "test_iter_log.csv", header)
    sum_path = log_dir / "summary_epoch.csv"
    sum_f = sum_path.open("w", encoding="utf-8", newline="")
    sum_f.write("epoch,train_loss,train_r2,test_loss,test_mae_kg,test_rmse_kg,test_mape,test_r2,lr_reg,lr_enc\n")
    sum_f.flush()

    # loss
    if args.loss == "huber":
        crit = nn.SmoothL1Loss(beta=1.0)
    elif args.loss == "mse":
        crit = nn.MSELoss()
    else:
        crit = nn.L1Loss()

    # -------------------------
    # 训练：分两类
    # A) 冻结 encoder（所有 sklearn + mlp freeze 都走：缓存特征 -> 训练）
    # B) 微调 encoder（仅 mlp）
    # -------------------------

    # ========== A) 冻结 encoder ==========
    if (args.finetune_encoder == 0):
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        feat_train_pt = feat_dir / "train_zcr.pt"
        feat_test_pt  = feat_dir / "test_zcr.pt"
        feat_stats_pt = feat_dir / "feat_stats.pt"

        # 1) 生成/加载缓存
        need_extract_features = args.rebuild_features or (not feat_train_pt.exists()) or (not feat_test_pt.exists())
        if not need_extract_features:
            pack_tr = torch.load(str(feat_train_pt), map_location="cpu")
            pack_te = torch.load(str(feat_test_pt), map_location="cpu")
            if ("paths" not in pack_tr) or ("paths" not in pack_te):
                print("Cached features missing paths, rebuilding...")
                need_extract_features = True

        if need_extract_features:
            print("Extracting cached features (freeze encoder)...")
            feat_loader_train = DataLoader(train_set, batch_size=max(1, min(args.batch_size, 32)), shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True, collate_fn=collate_z)
            feat_loader_test  = DataLoader(test_set,  batch_size=max(1, min(args.batch_size, 32)), shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True, collate_fn=collate_z)
            Xtr_t, ytr_t, train_paths = extract_features_zcr_to_cpu(encoder, feat_loader_train, device, latent_dim)
            Xte_t, yte_t, test_paths = extract_features_zcr_to_cpu(encoder, feat_loader_test,  device, latent_dim)
            torch.save({"X": Xtr_t, "y": ytr_t, "paths": train_paths}, str(feat_train_pt))
            torch.save({"X": Xte_t, "y": yte_t, "paths": test_paths}, str(feat_test_pt))
        else:
            print("Loading cached features...")
            pack_tr = torch.load(str(feat_train_pt), map_location="cpu")
            pack_te = torch.load(str(feat_test_pt), map_location="cpu")
            Xtr_t, ytr_t, train_paths = pack_tr["X"], pack_tr["y"], pack_tr["paths"]
            Xte_t, yte_t, test_paths = pack_te["X"], pack_te["y"], pack_te["paths"]

        if Xtr_t.numel() == 0:
            raise RuntimeError("训练特征为空：请检查 train.jsonl / root_dir / ply_path / label 是否有效")

        # 2) 特征 z-score（强烈建议开）
        feat_mean = Xtr_t.mean(dim=0, keepdim=True)
        feat_std  = Xtr_t.std(dim=0, keepdim=True).clamp_min(1e-6)
        torch.save({"feat_mean": feat_mean, "feat_std": feat_std}, str(feat_stats_pt))

        if args.debug_stats:
            print("[DEBUG] label mean/std:", y_mean, y_std)
            print("[DEBUG] X_train mean(min/max):", float(feat_mean.min()), float(feat_mean.max()))
            print("[DEBUG] X_train std (min/max):", float(feat_std.min()), float(feat_std.max()))
            print("[DEBUG] any NaN in Xtr:", bool(torch.isnan(Xtr_t).any().item()),
                  "any NaN in ytr:", bool(torch.isnan(ytr_t).any().item()))

        if args.feat_zscore:
            Xtr_t = (Xtr_t - feat_mean) / feat_std
            Xte_t = (Xte_t - feat_mean) / feat_std

        # 3) label z-score（训练空间）
        ytr = ytr_t.float()
        yte = yte_t.float()
        ytr_n = norm_y(ytr)
        yte_n = norm_y(yte)

        # 4) 选择回归器
        if args.reg_model == "mlp":
            reg = TorchMLPRegressor(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
            opt = torch.optim.AdamW(reg.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)

            train_ds = TensorDataset(Xtr_t.float(), ytr_n)
            test_ds  = TensorDataset(Xte_t.float(), yte_n)
            train_loader_feat = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
            test_loader_feat  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_train_loader_feat = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            for epoch in range(1, args.epochs + 1):
                reg.train()
                losses = []

                pbar = tqdm(train_loader_feat, desc=f"[MLP|FreezeEnc] epoch {epoch}/{args.epochs}", dynamic_ncols=True)
                for it, (feat, y_n) in enumerate(pbar):
                    feat = feat.to(device)
                    y_n  = y_n.to(device)

                    opt.zero_grad(set_to_none=True)
                    pred_n = reg(feat)
                    loss = crit(pred_n, y_n)
                    loss.backward()
                    opt.step()

                    losses.append(float(loss.detach().cpu()))
                    avg_loss = float(np.mean(losses))

                    # kg 指标
                    pred_kg = denorm_y(pred_n.detach())
                    y_kg = denorm_y(y_n.detach())
                    mae_kg = float(mae_t(pred_kg, y_kg).detach().cpu())
                    rmse_kg = float(rmse_t(pred_kg, y_kg).detach().cpu())
                    mape = float(mape_t(pred_kg, y_kg).detach().cpu())

                    pbar.set_postfix({"loss": f"{avg_loss:.5f}", "mae": f"{mae_kg:.3f}kg"})

                    train_logger.log({
                        "time": now_str(), "split": "train", "epoch": epoch, "iter": it,
                        "loss": float(loss.detach().cpu()),
                        "mae_kg": mae_kg, "rmse_kg": rmse_kg, "mape": mape, "r2": float("nan"),
                        "lr_reg": opt.param_groups[0]["lr"], "lr_enc": 0.0
                    })

                train_logger.flush()

                # eval：计算 test loss（同 crit，在 y_n 空间）+ kg 指标
                reg.eval()
                test_losses = []
                preds_all, ys_all = [], []
                with torch.no_grad():
                    for feat, y_n in test_loader_feat:
                        feat = feat.to(device)
                        y_n  = y_n.to(device)
                        pred_n = reg(feat)
                        test_losses.append(float(crit(pred_n, y_n).detach().cpu()))
                        preds_all.append(denorm_y(pred_n).detach().cpu())
                        ys_all.append(denorm_y(y_n).detach().cpu())

                test_loss = float(np.mean(test_losses)) if len(test_losses) else float("nan")
                pred = torch.cat(preds_all, dim=0)
                y = torch.cat(ys_all, dim=0)
                test_mae = float(mae_t(pred, y))
                test_rmse = float(rmse_t(pred, y))
                test_mape = float(mape_t(pred, y))
                test_r2 = float(r2_t(pred, y).detach().cpu())

                # train R2（epoch级：全训练集）
                preds_tr, ys_tr = [], []
                with torch.no_grad():
                    for feat, y_n in eval_train_loader_feat:
                        feat = feat.to(device)
                        y_n  = y_n.to(device)
                        pred_n = reg(feat)
                        preds_tr.append(denorm_y(pred_n).detach().cpu())
                        ys_tr.append(denorm_y(y_n).detach().cpu())
                if len(preds_tr):
                    pred_tr = torch.cat(preds_tr, dim=0)
                    y_tr = torch.cat(ys_tr, dim=0)
                    train_r2 = float(r2_t(pred_tr, y_tr).detach().cpu())
                else:
                    train_r2 = float("nan")

                test_logger.log({
                    "time": now_str(), "split": "test", "epoch": epoch, "iter": 0,
                    "loss": test_loss, "mae_kg": test_mae, "rmse_kg": test_rmse, "mape": test_mape, "r2": test_r2,
                    "lr_reg": opt.param_groups[0]["lr"], "lr_enc": 0.0
                })
                test_logger.flush()

                sum_f.write(f"{epoch},{float(np.mean(losses)):.8f},{train_r2:.8f},{test_loss:.8f},{test_mae:.8f},{test_rmse:.8f},{test_mape:.8f},{test_r2:.8f},"
                            f"{opt.param_groups[0]['lr']:.10f},0.0\n")
                sum_f.flush()

                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    best_epoch = epoch
                    best_model_path = ckpt_dir / "best_mlp.pth"

                    payload = {
                        "epoch": epoch,
                        "reg_state": reg.state_dict(),
                        "extra": {
                            "mode": "freeze_encoder_cached",
                            "reg_model": "mlp",
                            "hidden": args.hidden,
                            "dropout": args.dropout,
                            "label_mean": y_mean, "label_std": y_std, "label_zscore": bool(args.label_zscore),
                            "feat_zscore": bool(args.feat_zscore),
                            "feat_mean": feat_mean.squeeze(0).tolist(),
                            "feat_std":  feat_std.squeeze(0).tolist(),
                            "latent_dim": latent_dim, "in_dim": in_dim,
                            "ae_ckpt": args.ae_ckpt,
                            "best_metric": "test_mae_kg",
                            "best_metric_value": test_mae,
                        }
                    }
                    torch.save(payload, str(best_model_path))
                    save_prediction_table(pred_dir / "train_predictions_best.csv", train_paths, y_tr.numpy(), pred_tr.numpy())
                    save_prediction_table(pred_dir / "test_predictions_best.csv", test_paths, y.numpy(), pred.numpy())
                    save_best_info(best_info_path, {
                        "best_metric": "test_mae_kg",
                        "best_metric_value": test_mae,
                        "best_epoch": epoch,
                        "best_model_path": str(best_model_path),
                    })

                sch.step()
                print(f"[MLP|FreezeEnc] epoch {epoch:03d}: test_loss={test_loss:.6f}, test_mae={test_mae:.4f}kg, rmse={test_rmse:.4f}kg, mape={test_mape:.2f}%, test_r2={test_r2:.4f}, train_r2={train_r2:.4f}")

        else:
            # sklearn：用 numpy 训练
            Xtr = Xtr_t.numpy().astype(np.float32)
            Xte = Xte_t.numpy().astype(np.float32)
            ytr_np = ytr.numpy().astype(np.float32)
            yte_np = yte.numpy().astype(np.float32)

            # 训练目标（y空间）：zscore 可选
            if args.label_zscore:
                ytr_train = ((ytr_np - y_mean) / y_std).astype(np.float32)
                yte_eval_n = ((yte_np - y_mean) / y_std).astype(np.float32)
            else:
                ytr_train = ytr_np
                yte_eval_n = yte_np

            model = make_sklearn_model(args.reg_model, reg_params, seed=args.seed)

            # 对需要 scale 的模型，建议做一次 feature 标准化（我们已做 feat_zscore，所以这里不再额外加 StandardScaler）
            # 但如果你关了 feat_zscore，SVR/Ridge 等可能会很难训。

            if args.reg_model == "sgd":
                # epoch/iter 训练：partial_fit
                # sgd_regressor 需要先知道 y 的类型/范围，但回归不需要 classes
                N = Xtr.shape[0]
                bs = args.batch_size
                idxs = np.arange(N)

                for epoch in range(1, args.epochs + 1):
                    np.random.shuffle(idxs)
                    proxy_losses = []

                    pbar = tqdm(range(0, N, bs), desc=f"[SGD|FreezeEnc] epoch {epoch}/{args.epochs}", dynamic_ncols=True)
                    for it, st in enumerate(pbar):
                        sel = idxs[st:st+bs]
                        xb = Xtr[sel]
                        yb = ytr_train[sel]

                        # partial_fit（回归无需classes）
                        model.partial_fit(xb, yb)

                        # proxy：用 batch 上的 MAE(训练空间) 做一个 loss 记录
                        yb_pred = model.predict(xb).astype(np.float32)
                        proxy = float(np.mean(np.abs(yb_pred - yb)))
                        proxy_losses.append(proxy)

                        # kg 指标（更直观）
                        if args.label_zscore:
                            yb_pred_kg = yb_pred * y_std + y_mean
                            yb_kg = yb * y_std + y_mean
                        else:
                            yb_pred_kg = yb_pred
                            yb_kg = yb
                        mae_kg = float(np.mean(np.abs(yb_pred_kg - yb_kg)))
                        rmse_kg = float(np.sqrt(np.mean((yb_pred_kg - yb_kg) ** 2) + 1e-12))
                        mape = float(np.mean(np.abs(yb_pred_kg - yb_kg) / (np.abs(yb_kg) + 1e-6)) * 100.0)

                        pbar.set_postfix({"proxy": f"{proxy:.5f}", "mae": f"{mae_kg:.3f}kg"})

                        train_logger.log({
                            "time": now_str(), "split": "train", "epoch": epoch, "iter": it,
                            "loss": proxy,
                            "mae_kg": mae_kg, "rmse_kg": rmse_kg, "mape": mape, "r2": float("nan"),
                            "lr_reg": float("nan"), "lr_enc": 0.0
                        })

                    train_logger.flush()

                    # eval：计算 test_loss（训练空间 MSE） + kg 指标
                    y_pred_n = model.predict(Xte).astype(np.float32)
                    test_loss = mse_np(y_pred_n, yte_eval_n)  # 非NaN

                    # kg 指标
                    if args.label_zscore:
                        y_pred_kg = y_pred_n * y_std + y_mean
                    else:
                        y_pred_kg = y_pred_n

                    test_mae = float(np.mean(np.abs(y_pred_kg - yte_np)))
                    test_rmse = float(np.sqrt(np.mean((y_pred_kg - yte_np) ** 2) + 1e-12))
                    test_mape = float(np.mean(np.abs(y_pred_kg - yte_np) / (np.abs(yte_np) + 1e-6)) * 100.0)

                    # R2（kg空间）
                    y_pred_tr_n = model.predict(Xtr).astype(np.float32)
                    if args.label_zscore:
                        y_pred_tr_kg = y_pred_tr_n * y_std + y_mean
                    else:
                        y_pred_tr_kg = y_pred_tr_n
                    train_r2 = r2_np(y_pred_tr_kg, ytr_np)
                    test_r2 = r2_np(y_pred_kg, yte_np)

                    test_logger.log({
                        "time": now_str(), "split": "test", "epoch": epoch, "iter": 0,
                        "loss": test_loss, "mae_kg": test_mae, "rmse_kg": test_rmse, "mape": test_mape, "r2": test_r2,
                        "lr_reg": float("nan"), "lr_enc": 0.0
                    })
                    test_logger.flush()

                    sum_f.write(f"{epoch},nan,{train_r2:.8f},{test_loss:.8f},{test_mae:.8f},{test_rmse:.8f},{test_mape:.8f},{test_r2:.8f},nan,0.0\n")
                    sum_f.flush()

                    if test_mae < best_test_mae:
                        best_test_mae = test_mae
                        best_epoch = epoch
                        best_model_path = ckpt_dir / "best_sklearn.joblib"
                        sklearn_save({
                            "model": model,
                            "reg_model": args.reg_model,
                            "label_mean": y_mean, "label_std": y_std, "label_zscore": bool(args.label_zscore),
                            "feat_zscore": bool(args.feat_zscore),
                            "feat_mean": feat_mean.squeeze(0).numpy(),
                            "feat_std":  feat_std.squeeze(0).numpy(),
                            "ae_ckpt": args.ae_ckpt,
                            "best_metric": "test_mae_kg",
                            "best_metric_value": test_mae,
                            "best_epoch": epoch,
                        }, best_model_path)
                        save_prediction_table(pred_dir / "train_predictions_best.csv", train_paths, ytr_np, y_pred_tr_kg)
                        save_prediction_table(pred_dir / "test_predictions_best.csv", test_paths, yte_np, y_pred_kg)
                        save_best_info(best_info_path, {
                            "best_metric": "test_mae_kg",
                            "best_metric_value": test_mae,
                            "best_epoch": epoch,
                            "best_model_path": str(best_model_path),
                        })

                    print(f"[SGD|FreezeEnc] epoch {epoch:03d}: test_loss={test_loss:.6f}, test_mae={test_mae:.4f}kg, rmse={test_rmse:.4f}kg, mape={test_mape:.2f}%, test_r2={test_r2:.4f}, train_r2={train_r2:.4f}")

            else:
                # one-shot fit
                model.fit(Xtr, ytr_train)

                y_pred_n = model.predict(Xte).astype(np.float32)
                test_loss = mse_np(y_pred_n, yte_eval_n)  # 非NaN

                if args.label_zscore:
                    y_pred_kg = y_pred_n * y_std + y_mean
                else:
                    y_pred_kg = y_pred_n

                test_mae = float(np.mean(np.abs(y_pred_kg - yte_np)))
                test_rmse = float(np.sqrt(np.mean((y_pred_kg - yte_np) ** 2) + 1e-12))
                test_mape = float(np.mean(np.abs(y_pred_kg - yte_np) / (np.abs(yte_np) + 1e-6)) * 100.0)

                # R2（kg空间）
                y_pred_tr_n = model.predict(Xtr).astype(np.float32)
                if args.label_zscore:
                    y_pred_tr_kg = y_pred_tr_n * y_std + y_mean
                else:
                    y_pred_tr_kg = y_pred_tr_n
                train_r2 = r2_np(y_pred_tr_kg, ytr_np)
                test_r2 = r2_np(y_pred_kg, yte_np)

                # 记录一行（epoch=1）
                train_logger.log({
                    "time": now_str(), "split": "train", "epoch": 1, "iter": 0,
                    "loss": float("nan"), "mae_kg": float("nan"), "rmse_kg": float("nan"), "mape": float("nan"), "r2": train_r2,
                    "lr_reg": float("nan"), "lr_enc": 0.0
                })
                test_logger.log({
                    "time": now_str(), "split": "test", "epoch": 1, "iter": 0,
                    "loss": test_loss, "mae_kg": test_mae, "rmse_kg": test_rmse, "mape": test_mape, "r2": test_r2,
                    "lr_reg": float("nan"), "lr_enc": 0.0
                })
                train_logger.flush()
                test_logger.flush()

                sum_f.write(f"1,nan,{train_r2:.8f},{test_loss:.8f},{test_mae:.8f},{test_rmse:.8f},{test_mape:.8f},{test_r2:.8f},nan,0.0\n")
                sum_f.flush()

                best_test_mae = test_mae
                best_epoch = 1
                best_model_path = ckpt_dir / "best_sklearn.joblib"
                sklearn_save({
                    "model": model,
                    "reg_model": args.reg_model,
                    "label_mean": y_mean, "label_std": y_std, "label_zscore": bool(args.label_zscore),
                    "feat_zscore": bool(args.feat_zscore),
                    "feat_mean": feat_mean.squeeze(0).numpy(),
                    "feat_std":  feat_std.squeeze(0).numpy(),
                    "ae_ckpt": args.ae_ckpt,
                    "best_metric": "test_mae_kg",
                    "best_metric_value": test_mae,
                    "best_epoch": 1,
                }, best_model_path)
                save_prediction_table(pred_dir / "train_predictions_best.csv", train_paths, ytr_np, y_pred_tr_kg)
                save_prediction_table(pred_dir / "test_predictions_best.csv", test_paths, yte_np, y_pred_kg)
                save_best_info(best_info_path, {
                    "best_metric": "test_mae_kg",
                    "best_metric_value": test_mae,
                    "best_epoch": 1,
                    "best_model_path": str(best_model_path),
                })

                print(f"[{args.reg_model}|FreezeEnc] one-shot: test_loss={test_loss:.6f}, test_mae={test_mae:.4f}kg, rmse={test_rmse:.4f}kg, mape={test_mape:.2f}%, test_r2={test_r2:.4f}, train_r2={train_r2:.4f}")

    # ========== B) 微调 encoder（仅 mlp） ==========
    else:
        if args.reg_model != "mlp":
            raise ValueError("finetune_encoder=1 仅支持 reg_model=mlp")

        reg = TorchMLPRegressor(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)

        encoder.train()
        for p in encoder.parameters():
            p.requires_grad_(True)
        if args.encoder_bn_eval:
            set_batchnorm_eval(encoder)

        opt = torch.optim.AdamW(
            [{"params": reg.parameters(), "lr": args.lr},
             {"params": encoder.parameters(), "lr": args.enc_lr}],
            weight_decay=args.weight_decay
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=min(args.lr, args.enc_lr) * 0.1)

        # 估计一次 feat_mean/std（固定用，避免尺度乱）
        if args.feat_zscore:
            feat_mean, feat_std = estimate_feat_stats_welford(
                encoder=encoder,
                loader=DataLoader(train_set, batch_size=max(1, min(args.batch_size, 32)), shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=collate_z),
                device=device,
                latent_dim=latent_dim,
                max_samples=args.feat_stats_max_samples
            )
        else:
            feat_mean = torch.zeros(1, in_dim)
            feat_std = torch.ones(1, in_dim)

        if args.debug_stats:
            print("[DEBUG] finetune feat_mean(min/max):", float(feat_mean.min()), float(feat_mean.max()))
            print("[DEBUG] finetune feat_std (min/max):", float(feat_std.min()), float(feat_std.max()))

        # 用于 epoch 级 train R2 的评估 loader（shuffle=False）
        eval_loader_train = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_z
        )

        for epoch in range(1, args.epochs + 1):
            reg.train()
            encoder.train()
            if args.encoder_bn_eval:
                set_batchnorm_eval(encoder)

            losses = []
            pbar = tqdm(loader_train, desc=f"[MLP|FinetuneEnc] epoch {epoch}/{args.epochs}", dynamic_ncols=True)
            for it, batch in enumerate(pbar):
                if batch is None:
                    continue
                pts = batch["points"].to(device, non_blocking=True)
                cr = batch["cr"].to(device, non_blocking=True)
                y = batch["label"].to(device, non_blocking=True)
                y_n = norm_y(y)

                opt.zero_grad(set_to_none=True)

                z = encoder(pts)
                feat = torch.cat([z, cr], dim=1)
                if args.feat_zscore:
                    feat = (feat - feat_mean.to(device)) / feat_std.to(device)

                pred_n = reg(feat)
                loss = crit(pred_n, y_n)
                loss.backward()
                opt.step()

                losses.append(float(loss.detach().cpu()))
                avg_loss = float(np.mean(losses))

                pred_kg = denorm_y(pred_n.detach())
                mae_kg = float(mae_t(pred_kg, y).detach().cpu())
                rmse_kg = float(rmse_t(pred_kg, y).detach().cpu())
                mape = float(mape_t(pred_kg, y).detach().cpu())

                lr_reg = opt.param_groups[0]["lr"]
                lr_enc = opt.param_groups[1]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.5f}", "mae": f"{mae_kg:.3f}kg", "lrE": f"{lr_enc:.1e}"})

                train_logger.log({
                    "time": now_str(), "split": "train", "epoch": epoch, "iter": it,
                    "loss": float(loss.detach().cpu()),
                    "mae_kg": mae_kg, "rmse_kg": rmse_kg, "mape": mape, "r2": float("nan"),
                    "lr_reg": lr_reg, "lr_enc": lr_enc
                })
            train_logger.flush()

            # eval：test loss + kg 指标
            reg.eval()
            encoder.eval()
            test_losses = []
            preds_all, ys_all, test_paths_epoch = [], [], []
            feat_mean_dev = feat_mean.to(device)
            feat_std_dev = feat_std.to(device)
            with torch.no_grad():
                for batch in loader_test:
                    if batch is None:
                        continue
                    pts = batch["points"].to(device, non_blocking=True)
                    cr = batch["cr"].to(device, non_blocking=True)
                    y = batch["label"].to(device, non_blocking=True)
                    y_n = norm_y(y)

                    z = encoder(pts)
                    feat = torch.cat([z, cr], dim=1)
                    if args.feat_zscore:
                        feat = (feat - feat_mean_dev) / feat_std_dev

                    pred_n = reg(feat)
                    test_losses.append(float(crit(pred_n, y_n).detach().cpu()))
                    preds_all.append(denorm_y(pred_n).detach().cpu())
                    ys_all.append(y.detach().cpu())
                    metas = batch.get("meta", [])
                    for meta in metas:
                        if isinstance(meta, dict):
                            test_paths_epoch.append(str(meta.get("ply_path", "")))
                        else:
                            test_paths_epoch.append("")

            test_loss = float(np.mean(test_losses)) if len(test_losses) else float("nan")
            pred = torch.cat(preds_all, dim=0)
            y = torch.cat(ys_all, dim=0)
            test_mae = float(mae_t(pred, y))
            test_rmse = float(rmse_t(pred, y))
            test_mape = float(mape_t(pred, y))
            test_r2 = float(r2_t(pred, y).detach().cpu())

            # train R2（epoch级：全训练集）
            preds_tr, ys_tr, train_paths_epoch = [], [], []
            with torch.no_grad():
                for batch in eval_loader_train:
                    if batch is None:
                        continue
                    pts = batch["points"].to(device, non_blocking=True)
                    cr = batch["cr"].to(device, non_blocking=True)
                    ytr = batch["label"].to(device, non_blocking=True)

                    z = encoder(pts)
                    feat = torch.cat([z, cr], dim=1)
                    if args.feat_zscore:
                        feat = (feat - feat_mean_dev) / feat_std_dev

                    pred_n = reg(feat)
                    preds_tr.append(denorm_y(pred_n).detach().cpu())
                    ys_tr.append(ytr.detach().cpu())
                    metas = batch.get("meta", [])
                    for meta in metas:
                        if isinstance(meta, dict):
                            train_paths_epoch.append(str(meta.get("ply_path", "")))
                        else:
                            train_paths_epoch.append("")

            if len(preds_tr):
                pred_tr = torch.cat(preds_tr, dim=0)
                y_tr = torch.cat(ys_tr, dim=0)
                train_r2 = float(r2_t(pred_tr, y_tr).detach().cpu())
            else:
                train_r2 = float("nan")

            test_logger.log({
                "time": now_str(), "split": "test", "epoch": epoch, "iter": 0,
                "loss": test_loss, "mae_kg": test_mae, "rmse_kg": test_rmse, "mape": test_mape, "r2": test_r2,
                "lr_reg": opt.param_groups[0]["lr"], "lr_enc": opt.param_groups[1]["lr"]
            })
            test_logger.flush()

            sum_f.write(f"{epoch},{float(np.mean(losses)):.8f},{train_r2:.8f},{test_loss:.8f},{test_mae:.8f},{test_rmse:.8f},{test_mape:.8f},{test_r2:.8f},"
                        f"{opt.param_groups[0]['lr']:.10f},{opt.param_groups[1]['lr']:.10f}\n")
            sum_f.flush()

            if test_mae < best_test_mae:
                best_test_mae = test_mae
                best_epoch = epoch
                best_model_path = ckpt_dir / "best_mlp.pth"
                payload = {
                    "epoch": epoch,
                    "reg_state": reg.state_dict(),
                    "enc_state": encoder.state_dict(),
                    "extra": {
                        "mode": "finetune_encoder",
                        "reg_model": "mlp",
                        "hidden": args.hidden,
                        "dropout": args.dropout,
                        "label_mean": y_mean, "label_std": y_std, "label_zscore": bool(args.label_zscore),
                        "feat_zscore": bool(args.feat_zscore),
                        "feat_mean": feat_mean.squeeze(0).tolist(),
                        "feat_std":  feat_std.squeeze(0).tolist(),
                        "latent_dim": latent_dim, "in_dim": in_dim,
                        "ae_ckpt": args.ae_ckpt,
                        "encoder_bn_eval": bool(args.encoder_bn_eval),
                        "best_metric": "test_mae_kg",
                        "best_metric_value": test_mae,
                    }
                }
                torch.save(payload, str(best_model_path))
                save_prediction_table(pred_dir / "train_predictions_best.csv", train_paths_epoch, y_tr.numpy(), pred_tr.numpy())
                save_prediction_table(pred_dir / "test_predictions_best.csv", test_paths_epoch, y.numpy(), pred.numpy())
                save_best_info(best_info_path, {
                    "best_metric": "test_mae_kg",
                    "best_metric_value": test_mae,
                    "best_epoch": epoch,
                    "best_model_path": str(best_model_path),
                })

            sch.step()
            print(f"[MLP|FinetuneEnc] epoch {epoch:03d}: test_loss={test_loss:.6f}, test_mae={test_mae:.4f}kg, rmse={test_rmse:.4f}kg, mape={test_mape:.2f}%, test_r2={test_r2:.4f}, train_r2={train_r2:.4f}")

    # -------------------------
    # 收尾：写 Excel
    # -------------------------
    train_logger.close()
    test_logger.close()
    sum_f.close()

    excel_path = run_dir / "weight_regression_logs.xlsx"
    df_train = pd.read_csv(log_dir / "train_iter_log.csv")
    df_test = pd.read_csv(log_dir / "test_iter_log.csv")
    df_sum = pd.read_csv(sum_path)

    train_pred_csv = pred_dir / "train_predictions_best.csv"
    test_pred_csv = pred_dir / "test_predictions_best.csv"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_train.to_excel(writer, sheet_name="train_iter", index=False)
        df_test.to_excel(writer, sheet_name="test_epoch", index=False)
        df_sum.to_excel(writer, sheet_name="summary_epoch", index=False)
        if train_pred_csv.exists():
            pd.read_csv(train_pred_csv).to_excel(writer, sheet_name="train_pred_best", index=False)
        if test_pred_csv.exists():
            pd.read_csv(test_pred_csv).to_excel(writer, sheet_name="test_pred_best", index=False)

    print("All done.")
    print(f"Best epoch: {best_epoch}, best test_mae_kg: {best_test_mae:.6f}")
    if best_model_path is not None:
        print(f"Best model saved to: {best_model_path}")
    print(f"Prediction tables saved in: {pred_dir}")
    print(f"Outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()
