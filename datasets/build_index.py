import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def _read_excel_mapping(excel_path: Path, folder_col: str, label_col: str) -> List[Tuple[str, float]]:
    df = pd.read_excel(excel_path)

    # 支持传列名或列序号（如 "0"/"1"）
    def pick_col(col: str):
        if col.isdigit():
            return df.iloc[:, int(col)]
        return df[col]

    folders = pick_col(folder_col)
    labels = pick_col(label_col)

    pairs = []
    for f, y in zip(folders, labels):
        if pd.isna(f):
            continue
        folder_name = str(f).strip()
        if folder_name == "":
            continue
        label = float(y) if not pd.isna(y) else float("nan")
        pairs.append((folder_name, label))
    return pairs


def _find_capture_roots(folderA_dir: Path, max_depth: int = 6) -> List[Path]:
    """
    找到所有包含 PLY 文件夹的“采集根目录”（即 PLY 的父目录）。
    兼容两种情况：
      1) folderA_dir/PLY 直接存在
      2) folderA_dir/**/PLY 存在（PLY 的父目录就是 capture_root）
    """
    roots = []
    if (folderA_dir / "PLY").is_dir():
        roots.append(folderA_dir)

    # 递归搜索 **/PLY，但限制深度，避免扫太深太慢
    for ply_dir in folderA_dir.rglob("PLY"):
        if not ply_dir.is_dir():
            continue
        try:
            rel_parts = ply_dir.relative_to(folderA_dir).parts
        except Exception:
            continue
        if len(rel_parts) > max_depth:
            continue
        roots.append(ply_dir.parent)

    # 去重
    uniq = []
    seen = set()
    for r in roots:
        rp = str(r.resolve())
        if rp not in seen:
            uniq.append(r)
            seen.add(rp)
    return uniq


def _index_images(rgb_dir: Path) -> Dict[str, Path]:
    """stem -> image_path"""
    mapping = {}
    if not rgb_dir.is_dir():
        return mapping
    for p in rgb_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMG_EXTS:
            mapping[p.stem] = p
    return mapping


def _pair_ply_rgb(
    ply_files: List[Path],
    rgb_map: Dict[str, Path],
    rgb_dir: Optional[Path] = None,
) -> Tuple[List[Optional[Path]], List[bool]]:
    """
    返回：
      rgb_paths: 与 ply_files 一一对应的 rgb 路径（可能为 None）
      reliable: 是否“可靠匹配”（同 stem 匹配为 True，按排序对齐为 False）
    """
    rgb_paths: List[Optional[Path]] = []
    reliable: List[bool] = []

    # 先按 stem 精确匹配
    for pf in ply_files:
        rp = rgb_map.get(pf.stem, None)
        rgb_paths.append(rp)
        reliable.append(rp is not None)

    # 如果存在缺失，但 rgb_dir 下图片数量与 ply 数量相同，则尝试按排序对齐补全
    if rgb_dir is not None and any(p is None for p in rgb_paths):
        imgs = []
        if rgb_dir.is_dir():
            for p in rgb_dir.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    imgs.append(p)
        imgs = sorted(imgs, key=lambda x: x.name)
        if len(imgs) == len(ply_files):
            ply_sorted = sorted(ply_files, key=lambda x: x.name)
            # 逐个按排序对齐（标记为不可靠匹配）
            name_to_img = {p.name: imgs[i] for i, p in enumerate(ply_sorted)}
            rgb_paths = [name_to_img.get(p.name, None) for p in ply_files]
            reliable = [False] * len(ply_files)

    return rgb_paths, reliable


def _resolve_folderA_dirs(root_dir: Path, folderA: str, fallback_depth: int = 3) -> List[Path]:
    """
    返回所有匹配的 A 文件夹路径：
      1) root/A
      2) root/*/A   (aa 层)
      3) 找不到时，再在 root 下做一次 depth 限制的递归搜索兜底
    """
    hits: List[Path] = []

    # 1) root/A
    p0 = root_dir / folderA
    if p0.is_dir():
        hits.append(p0)

    # 2) root/*/A
    for aa in root_dir.iterdir():
        if aa.is_dir():
            p1 = aa / folderA
            if p1.is_dir():
                hits.append(p1)

    # 去重
    uniq = []
    seen = set()
    for h in hits:
        rp = str(h.resolve())
        if rp not in seen:
            uniq.append(h)
            seen.add(rp)
    hits = uniq

    # 3) fallback：浅层递归兜底（避免扫太深）
    if len(hits) == 0:
        for cand in root_dir.rglob(folderA):
            if not cand.is_dir():
                continue
            try:
                rel = cand.relative_to(root_dir)
                if len(rel.parts) <= fallback_depth:
                    hits.append(cand)
            except Exception:
                pass

        # 再去重
        uniq = []
        seen = set()
        for h in hits:
            rp = str(h.resolve())
            if rp not in seen:
                uniq.append(h)
                seen.add(rp)
        hits = uniq

    return hits


def build_index(
    root_dir: Path,
    excel_path: Path,
    out_dir: Path,
    folder_col: str,
    label_col: str,
    test_folders: List[str],
    rgb_prefer: List[str],
    max_depth: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"

    pairs = _read_excel_mapping(excel_path, folder_col, label_col)
    test_set = set([x.strip() for x in test_folders if x.strip() != ""])

    n_train = n_test = 0
    n_missing_A = 0
    n_missing_rgb = 0
    n_total = 0

    f_train = train_path.open("w", encoding="utf-8")
    f_test = test_path.open("w", encoding="utf-8")

    for folderA, label in tqdm(pairs, desc="Building index"):
        folderA_dirs = _resolve_folderA_dirs(root_dir, folderA, fallback_depth=max_depth + 2)
        if len(folderA_dirs) == 0:
            n_missing_A += 1
            continue

        split = "test" if folderA in test_set else "train"
        fout = f_test if split == "test" else f_train

        # 若同名 A 在不同 aa 下都存在：这里会全部收录
        for folderA_dir in folderA_dirs:
            capture_roots = _find_capture_roots(folderA_dir, max_depth=max_depth)
            if len(capture_roots) == 0:
                continue

            for cap_root in capture_roots:
                ply_dir = cap_root / "PLY"
                if not ply_dir.is_dir():
                    continue

                ply_files = sorted(
                    [p for p in ply_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ply"],
                    key=lambda x: x.name
                )
                if len(ply_files) == 0:
                    continue

                # 选择 RGB 文件夹（按 prefer 顺序）
                rgb_dir = None
                for cand in rgb_prefer:
                    d = cap_root / cand
                    if d.is_dir():
                        rgb_dir = d
                        break

                rgb_map = _index_images(rgb_dir) if rgb_dir is not None else {}
                rgb_paths, reliable = _pair_ply_rgb(ply_files, rgb_map, rgb_dir=rgb_dir)

                for pf, ip, ok in zip(ply_files, rgb_paths, reliable):
                    rec = {
                        "split": split,
                        "folderA": folderA,
                        "aa": folderA_dir.parent.name,  # A 的上一级（aa）
                        "folderA_path": str(folderA_dir.relative_to(root_dir)),
                        "label": label,
                        "capture_root": str(cap_root.relative_to(root_dir)),
                        "ply_path": str(pf.relative_to(root_dir)),
                        "rgb_path": str(ip.relative_to(root_dir)) if ip is not None else None,
                        "rgb_reliable": bool(ok),
                    }

                    if rec["rgb_path"] is None:
                        n_missing_rgb += 1

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_total += 1
                    if split == "test":
                        n_test += 1
                    else:
                        n_train += 1

    f_train.close()
    f_test.close()

    print("\nIndex done.")
    print(f"  root_dir      : {root_dir}")
    print(f"  excel         : {excel_path}")
    print(f"  out_dir       : {out_dir}")
    print(f"  total samples : {n_total}")
    print(f"  train samples : {n_train}")
    print(f"  test samples  : {n_test}")
    print(f"  missing A dirs: {n_missing_A}")
    print(f"  missing rgb   : {n_missing_rgb}")
    print(f"  train index   : {train_path}")
    print(f"  test index    : {test_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, required=True, help="数据根目录（包含 aa 文件夹 或 直接包含 A 文件夹）")
    ap.add_argument("--excel", type=str, required=True, help="Excel 路径：A 列 + label 列")
    ap.add_argument("--out_dir", type=str, default="indicestest", help="输出索引目录")
    ap.add_argument("--folder_col", type=str, default="0", help="A 列：列名 或 列序号(默认0)")
    ap.add_argument("--label_col", type=str, default="1", help="label 列：列名 或 列序号(默认1)")
    ap.add_argument("--test_folders", type=str, default="", help="测试集 A 名称，用逗号分隔：A1,A2,...")
    ap.add_argument("--test_list_file", type=str, default="", help="测试集 A 名称 txt，每行一个（可选）")
    ap.add_argument("--rgb_prefer", type=str, default="RGBnew,RGB",
                    help="RGB 选择顺序，用逗号分隔。默认优先 RGBnew，其次 RGB")
    ap.add_argument("--max_depth", type=int, default=6, help="在 A 文件夹内搜索 PLY 的最大深度")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root_dir)
    excel_path = Path(args.excel)
    out_dir = Path(args.out_dir)

    test_folders = []
    if args.test_folders.strip():
        test_folders += [x.strip() for x in args.test_folders.split(",") if x.strip()]
    if args.test_list_file.strip():
        p = Path(args.test_list_file)
        if p.is_file():
            test_folders += [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

    rgb_prefer = [x.strip() for x in args.rgb_prefer.split(",") if x.strip()]

    build_index(
        root_dir=root_dir,
        excel_path=excel_path,
        out_dir=out_dir,
        folder_col=args.folder_col,
        label_col=args.label_col,
        test_folders=test_folders,
        rgb_prefer=rgb_prefer,
        max_depth=args.max_depth,
    )
