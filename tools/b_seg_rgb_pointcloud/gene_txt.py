# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path


def iter_rgb_folders(root: Path):
    """递归找到所有名为 RGB 的文件夹（大小写不敏感）"""
    for dirpath, dirnames, filenames in os.walk(root):
        if Path(dirpath).name.lower() == "rgb":
            yield Path(dirpath)


def list_jpgs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def save_yolo_seg_txt_one(result, txt_path: Path):
    """
    保存单张图的 YOLO-seg 标注：
    每行：cls x1 y1 x2 y2 ...（x,y均为0~1）
    """
    # 没检测到实例：写空文件（和训练常规一致）
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        txt_path.write_text("", encoding="utf-8")
        return 0

    classes = result.boxes.cls
    if classes is None:
        txt_path.write_text("", encoding="utf-8")
        return 0

    # xyn: list[np.ndarray], 每个实例一个多边形，点坐标已归一化到0~1
    polys = result.masks.xyn
    if polys is None or len(polys) == 0:
        txt_path.write_text("", encoding="utf-8")
        return 0

    # 有时 boxes 和 masks 数量不一致，取最小长度对齐
    n = min(len(classes), len(polys))
    lines = []

    for i in range(n):
        cls_id = int(classes[i].item())
        poly = polys[i]
        if poly is None or len(poly) < 3:
            continue

        # 展平为 x1 y1 x2 y2 ...
        coords = " ".join([f"{v:.6f}" for xy in poly for v in xy])
        lines.append(f"{cls_id} {coords}")

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default=r"./runs/segment/s/weights/best.pt",
                    help="训练好的 best.pt 路径")
    ap.add_argument("--root", type=str, default=r"/2024219001/data/handheld_pigweight/data_zhj", help="数据根目录，递归寻找 RGB 文件夹")
    ap.add_argument("--rgb_folder_name", type=str, default="RGB", help="RGB文件夹名称（默认RGB，大小写不敏感）")

    ap.add_argument("--imgsz", type=int, default=640, help="推理尺寸 imgsz（默认640）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值 conf（默认0.25）")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS iou阈值（默认0.7）")
    ap.add_argument("--device", type=str, default="0", help="设备，例如 '0' / 'cpu'，默认留空=自动")
    ap.add_argument("--half", action="store_true", help="启用FP16（仅GPU有效）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的txt（默认不覆盖）")

    args = ap.parse_args()

    weights = Path(args.weights)
    root = Path(args.root)

    if not weights.exists():
        raise FileNotFoundError(f"找不到权重文件：{weights}")
    if not root.exists():
        raise FileNotFoundError(f"找不到root目录：{root}")

    # 兼容：优先用 pip 的 ultralytics；如果你想强制用仓库版，请在运行前把 ultralytics-main 加到 PYTHONPATH
    from ultralytics import YOLO

    model = YOLO(str(weights))

    total_imgs = 0
    total_txt = 0
    total_instances = 0

    # 遍历所有 RGB 文件夹
    for rgb_dir in iter_rgb_folders(root):
        if rgb_dir.name.lower() != args.rgb_folder_name.lower():
            continue

        jpgs = list_jpgs(rgb_dir)
        if not jpgs:
            continue

        # 过滤掉不需要处理的（已有txt且不覆盖）
        tasks = []
        for img_path in jpgs:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists() and (not args.overwrite):
                continue
            tasks.append((img_path, txt_path))

        if not tasks:
            continue

        img_paths = [str(p[0]) for p in tasks]
        total_imgs += len(img_paths)

        # 批量推理（同一文件夹内批处理速度更好）
        results = model.predict(
            source=img_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device.strip() else None,
            half=args.half,
            verbose=False
        )

        # 保存txt
        for (img_path, txt_path), res in zip(tasks, results):
            n_inst = save_yolo_seg_txt_one(res, txt_path)
            total_txt += 1
            total_instances += n_inst

        print(f"[完成] {rgb_dir}  处理图片 {len(img_paths)} 张")

    print("\n=== 汇总 ===")
    print(f"处理图片数：{total_imgs}")
    print(f"生成/更新txt：{total_txt}")
    print(f"输出实例总数（txt行数累计）：{total_instances}")
    print("全部txt与jpg同目录同名保存。")


if __name__ == "__main__":
    main()
