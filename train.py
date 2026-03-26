# train.py
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 让 “python train.py” 直接能 import 本项目模块
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from models import PointCloudAE
from datasets.plyae_dataset import PLYAutoEncoderDataset

from losses.chamfer import chamfer_distance
from losses.repulsion import repulsion_loss
from losses.knn_smooth import knn_edge_length_loss
from losses.hausdorff import hausdorff_distance
from losses.emd_sinkhorn import sinkhorn_emd


# -------------------------
# utils
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


class RetryDataset(Dataset):
    """
    包装一个可能返回 None 的 dataset：
      - 若 __getitem__ 返回 None / points=None，则随机换 index 重试若干次
      - 若一直失败，返回一个标记坏样本（points=None），交给 collate 过滤
    """

    def __init__(self, base: Dataset, max_retry: int = 20, seed: int = 0):
        self.base = base
        self.max_retry = int(max_retry)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        n = len(self.base)
        cur = idx
        for _ in range(self.max_retry):
            sample = self.base[cur]
            if isinstance(sample, dict):
                pts = sample.get("points", None)
                if isinstance(pts, torch.Tensor) and pts.numel() > 0:
                    return sample
            # sample is None or bad
            cur = int(self.rng.integers(0, n))
        # 兜底：返回坏样本（points None）
        return {"points": None, "label": float("nan"), "meta": {"bad": True, "orig_idx": idx}}


def collate_ae(batch: List[Any]) -> Optional[Dict[str, Any]]:
    """
    过滤掉 None / points=None 的样本，避免 default_collate 报错。
    若过滤后 batch 为空，返回 None（训练循环会跳过）。
    """
    valid = []
    for b in batch:
        if not isinstance(b, dict):
            continue
        pts = b.get("points", None)
        if isinstance(pts, torch.Tensor) and pts.ndim == 2 and pts.shape[1] == 3 and pts.numel() > 0:
            valid.append(b)

    if len(valid) == 0:
        return None

    out: Dict[str, Any] = {}
    out["points"] = torch.stack([b["points"] for b in valid], dim=0)  # (B,N,3)

    labels = [b.get("label", float("nan")) for b in valid]
    out["label"] = torch.tensor(labels, dtype=torch.float32)

    out["meta"] = [b.get("meta", {}) for b in valid]

    # 保留 rgb（如果你 Dataset 里有这个 key，哪怕是 None 也不影响）
    if "rgb" in valid[0]:
        out["rgb"] = [b.get("rgb", None) for b in valid]
    return out


class CSVLogger:
    """把每个 iter 的日志流式写入 CSV。"""

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

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.flush()
        self._f.close()


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, args_dict: dict):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "args": args_dict,
    }
    torch.save(ckpt, str(path))


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                    map_location="cpu"):
    ckpt = torch.load(str(path), map_location=map_location)
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch, ckpt


def build_amp(device: torch.device, amp_enabled: bool):
    """兼容新旧 AMP API。"""
    use_amp = bool(amp_enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and hasattr(torch.amp, "autocast"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def autocast_ctx():
            return torch.amp.autocast("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        def autocast_ctx():
            return torch.cuda.amp.autocast(enabled=use_amp)
    return scaler, autocast_ctx, use_amp


def compute_losses(
        recon: torch.Tensor,
        target: torch.Tensor,
        loss_cd_p: int = 2,
        w_rep: float = 0.05,
        rep_k: int = 8,
        rep_h: float = 0.03,
        w_knn: float = 0.01,
        knn_k: int = 8,
        w_hd: float = 0.0,
        w_emd: float = 0.0,
        emd_eps: float = 0.05,
        emd_iters: int = 30,
):
    cd = chamfer_distance(recon, target, p=loss_cd_p)

    rep = torch.tensor(0.0, device=recon.device)
    if w_rep > 0:
        rep = repulsion_loss(recon, k=rep_k, h=rep_h)

    knn = torch.tensor(0.0, device=recon.device)
    if w_knn > 0:
        knn = knn_edge_length_loss(recon, k=knn_k)

    hd = torch.tensor(0.0, device=recon.device)
    if w_hd > 0:
        hd = hausdorff_distance(recon, target, squared=True)

    emd = torch.tensor(0.0, device=recon.device)
    if w_emd > 0:
        emd = sinkhorn_emd(recon, target, eps=emd_eps, iters=emd_iters)

    total = cd + w_rep * rep + w_knn * knn + w_hd * hd + w_emd * emd

    items = {
        "cd": float(cd.detach().cpu()),
        "rep": float(rep.detach().cpu()) if w_rep > 0 else 0.0,
        "knn": float(knn.detach().cpu()) if w_knn > 0 else 0.0,
        "hd": float(hd.detach().cpu()) if w_hd > 0 else 0.0,
        "emd": float(emd.detach().cpu()) if w_emd > 0 else 0.0,
        "total": float(total.detach().cpu()),
    }
    return total, items


@torch.no_grad()
def run_eval(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        epoch: int,
        global_step_start: int,
        logger: CSVLogger,
        batch_size: int,
        args,
):
    model.eval()
    running = []
    global_step = global_step_start
    skipped = 0

    pbar = tqdm(loader, desc=f"[Test ] epoch {epoch}", leave=False, dynamic_ncols=True)
    for it, batch in enumerate(pbar):
        if batch is None:
            skipped += 1
            continue

        pts = batch["points"].to(device, non_blocking=True)
        recon, _ = model(pts)

        total, items = compute_losses(
            recon, pts,
            loss_cd_p=args.loss_cd_p,
            w_rep=args.w_rep,
            rep_k=args.rep_k,
            rep_h=args.rep_h,
            w_knn=args.w_knn,
            knn_k=args.knn_k,
            w_hd=args.w_hd,
            w_emd=0.0,  # eval 默认不算 EMD
        )

        running.append(items["total"])
        avg = float(np.mean(running)) if running else float("nan")

        pbar.set_postfix({"loss": f"{avg:.5f}", "cd": f"{items['cd']:.5f}", "skip": skipped})

        logger.log({
            "time": now_str(),
            "split": "test",
            "epoch": epoch,
            "iter": it,
            "global_step": global_step,
            "loss_total": items["total"],
            "loss_cd": items["cd"],
            "loss_rep": items["rep"],
            "loss_knn": items["knn"],
            "loss_hd": items["hd"],
            "loss_emd": 0.0,
            "lr": args.lr,
            "batch_size": batch_size,
        })
        global_step += 1

    logger.flush()
    return (float(np.mean(running)) if len(running) else float("nan")), global_step, skipped


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # 数据
    ap.add_argument("--root_dir", type=str, default="E:\data_zhj_new")
    ap.add_argument("--train_index", type=str, default="indicestest/train.jsonl")
    ap.add_argument("--test_index", type=str, default="indicestest/test.jsonl")

    # 输出
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default="")

    # 训练
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)  # 支持设为 1
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--drop_last_train", type=int, default=1)

    # 关键：遇到坏样本的重试次数（很有用）
    ap.add_argument("--max_retry", type=int, default=5)

    # 模型
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--width_mult", type=float, default=1.0)
    ap.add_argument("--enc_dropout", type=float, default=0.0)
    ap.add_argument("--use_bn", type=int, default=1)

    # Dataset
    ap.add_argument("--sample_mode", type=str, default="fps", choices=["fps", "random"])
    ap.add_argument("--fps_pre_n", type=int, default=8192)
    ap.add_argument("--normalize", type=int, default=1)

    # Loss
    ap.add_argument("--loss_cd_p", type=int, default=2, choices=[1, 2])
    ap.add_argument("--w_rep", type=float, default=0.05)
    ap.add_argument("--rep_k", type=int, default=8)
    ap.add_argument("--rep_h", type=float, default=0.03)
    ap.add_argument("--w_knn", type=float, default=0.002)
    ap.add_argument("--knn_k", type=int, default=8)
    ap.add_argument("--w_hd", type=float, default=0.0)

    # 低频 EMD（很慢，默认不开）
    ap.add_argument("--w_emd", type=float, default=0.0)
    ap.add_argument("--emd_every", type=int, default=5)
    ap.add_argument("--emd_eps", type=float, default=0.05)
    ap.add_argument("--emd_iters", type=int, default=30)

    # AMP
    ap.add_argument("--amp", type=int, default=1)

    args = ap.parse_args()
    args.use_bn = bool(args.use_bn)
    args.normalize = bool(args.normalize)
    args.amp = bool(args.amp)
    args.drop_last_train = bool(args.drop_last_train)

    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # run dir
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / (args.run_name.strip() if args.run_name.strip() else datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Run dir: {run_dir}")
    print(f"Device : {device}")

    # datasets (训练不返回 rgb)
    base_train = PLYAutoEncoderDataset(
        index_jsonl=args.train_index,
        root_dir=args.root_dir,
        n_points=args.n_points,
        sample_mode=args.sample_mode,
        fps_pre_n=args.fps_pre_n,
        normalize=args.normalize,
        return_rgb=False,
        rgb_to_tensor=False,
        seed=args.seed,
    )
    base_test = PLYAutoEncoderDataset(
        index_jsonl=args.test_index,
        root_dir=args.root_dir,
        n_points=args.n_points,
        sample_mode=args.sample_mode,
        fps_pre_n=args.fps_pre_n,
        normalize=args.normalize,
        return_rgb=False,
        rgb_to_tensor=False,
        seed=args.seed + 123,
    )

    # 包装：自动重试坏样本
    train_set = RetryDataset(base_train, max_retry=args.max_retry, seed=args.seed)
    test_set = RetryDataset(base_test, max_retry=args.max_retry, seed=args.seed + 999)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=args.drop_last_train,
        collate_fn=collate_ae,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_ae,
    )

    # model
    model = PointCloudAE(
        n_points=args.n_points,
        latent_dim=args.latent_dim,
        width_mult=args.width_mult,
        enc_dropout=args.enc_dropout,
        use_bn=args.use_bn,
        encoder_type="pointnet"
    ).to(device)

    print(f"Model params: {model.num_parameters()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    scaler, autocast_ctx, use_amp = build_amp(device, args.amp)
    print(f"AMP enabled: {use_amp}")

    # loggers
    header = [
        "time", "split", "epoch", "iter", "global_step",
        "loss_total", "loss_cd", "loss_rep", "loss_knn", "loss_hd", "loss_emd",
        "lr", "batch_size"
    ]
    train_logger = CSVLogger(log_dir / "train_iter_log.csv", header)
    test_logger = CSVLogger(log_dir / "test_iter_log.csv", header)
    summary_csv = log_dir / "summary_epoch.csv"
    summary_f = summary_csv.open("w", encoding="utf-8", newline="")
    summary_f.write("epoch,train_loss_avg,test_loss_avg,lr,train_skipped,test_skipped\n")
    summary_f.flush()

    # resume
    start_epoch = 1
    global_train_step = 0
    global_test_step = 0
    if args.resume.strip():
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            start_epoch, _ = load_checkpoint(ckpt_path, model, optimizer, map_location=device)
            print(f"Resumed from {ckpt_path}, start_epoch={start_epoch}")

    # train loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = []
        skipped = 0

        pbar = tqdm(train_loader, desc=f"[Train] epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for it, batch in enumerate(pbar):
            if batch is None:
                skipped += 1
                continue

            pts = batch["points"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                recon, _ = model(pts)

                use_emd_now = (args.w_emd > 0) and (global_train_step % max(1, args.emd_every) == 0)
                loss, items = compute_losses(
                    recon, pts,
                    loss_cd_p=args.loss_cd_p,
                    w_rep=args.w_rep,
                    rep_k=args.rep_k,
                    rep_h=args.rep_h,
                    w_knn=args.w_knn,
                    knn_k=args.knn_k,
                    w_hd=args.w_hd,
                    w_emd=(args.w_emd if use_emd_now else 0.0),
                    emd_eps=args.emd_eps,
                    emd_iters=args.emd_iters,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running.append(items["total"])
            avg = float(np.mean(running)) if running else float("nan")

            pbar.set_postfix({
                "loss": f"{avg:.5f}",
                "cd": f"{items['cd']:.5f}",
                "rep": f"{items['rep']:.5f}",
                "knn": f"{items['knn']:.5f}",
                "hd": f"{items['hd']:.5f}",
                "emd": f"{items['emd']:.5f}",
                "skip": skipped,
            })

            train_logger.log({
                "time": now_str(),
                "split": "train",
                "epoch": epoch,
                "iter": it,
                "global_step": global_train_step,
                "loss_total": items["total"],
                "loss_cd": items["cd"],
                "loss_rep": items["rep"],
                "loss_knn": items["knn"],
                "loss_hd": items["hd"],
                "loss_emd": items["emd"],
                "lr": optimizer.param_groups[0]["lr"],
                "batch_size": args.batch_size,
            })
            global_train_step += 1

        train_logger.flush()

        # eval
        test_avg, global_test_step, test_skipped = run_eval(
            model=model,
            loader=test_loader,
            device=device,
            epoch=epoch,
            global_step_start=global_test_step,
            logger=test_logger,
            batch_size=args.batch_size,
            args=args,
        )

        train_avg = float(np.mean(running)) if len(running) else float("nan")
        lr_now = optimizer.param_groups[0]["lr"]

        summary_f.write(f"{epoch},{train_avg:.8f},{test_avg:.8f},{lr_now:.10f},{skipped},{test_skipped}\n")
        summary_f.flush()

        scheduler.step()

        # save checkpoints
        save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, epoch, vars(args))
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, vars(args))

        print(f"Epoch {epoch:03d} done. train_avg={train_avg:.6f}, test_avg={test_avg:.6f}, "
              f"lr={lr_now:.6e}, train_skip={skipped}, test_skip={test_skipped}")

    train_logger.close()
    test_logger.close()
    summary_f.close()

    # write Excel
    excel_path = run_dir / "loss_curves.xlsx"
    print(f"Writing Excel: {excel_path}")

    df_train = pd.read_csv(log_dir / "train_iter_log.csv")
    df_test = pd.read_csv(log_dir / "test_iter_log.csv")
    df_sum = pd.read_csv(log_dir / "summary_epoch.csv")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_train.to_excel(writer, sheet_name="train_iter", index=False)
        df_test.to_excel(writer, sheet_name="test_iter", index=False)
        df_sum.to_excel(writer, sheet_name="summary_epoch", index=False)

    print("All done.")
    print(f"Run outputs saved in: {run_dir}")


if __name__ == "__main__":
    main()
