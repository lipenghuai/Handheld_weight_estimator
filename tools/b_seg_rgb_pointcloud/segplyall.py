# -*- coding: utf-8 -*-
# pip install open3d opencv-python numpy
import os
import json
import numpy as np
import cv2
import open3d as o3d

# 相机参数文件路径（Intel RealSense D435i）
cam_json_path = r"./lip/data/d435i_dump.json"

# 输入与输出主目录
input_base_dir = r"/2024219001/data/handheld_pigweight/data_zhj"
output_base_dir = r"/2024219001/data/handheld_pigweight/data"

# 日志文件路径
log_path = r"./pp.txt"

def load_d435i_params(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cs = cfg.get("chosen_streams", {})
    extr = cfg.get("extrinsics", {})
    # 读取彩色相机内参（优先使用color_intrinsics）
    c_intr = cs.get("color_intrinsics") or (cs.get("color_profile", {}).get("intrinsics"))
    if c_intr is None:
        raise KeyError("JSON中缺少 color_intrinsics 或 color_profile.intrinsics 字段")
    W = int(c_intr["width"]); H = int(c_intr["height"])
    fx = float(c_intr["fx"]); fy = float(c_intr["fy"])
    cx = float(c_intr["ppx"]); cy = float(c_intr["ppy"])
    # 深度坐标系到彩色坐标系的外参
    d2c = extr.get("depth_to_color")
    if d2c is None:
        raise KeyError("JSON中缺少 extrinsics.depth_to_color 字段")
    R = np.array(d2c["rotation_row_major_3x3"], dtype=np.float64).reshape(3, 3)
    t = np.array(d2c["translation_m"], dtype=np.float64).reshape(3,)
    return W, H, fx, fy, cx, cy, R, t

def parse_yolo_txt_to_mask(txt_path: str, W: int, H: int, keep_classes=None):
    """
    支持两种YOLO标注格式:
      1) 实例分割: <cls> <x1> <y1> <x2> <y2> ...  （多边形顶点，归一化坐标）
      2) 检测框:   <cls> <cx> <cy> <w> <h>        （中心点及宽高，归一化坐标）
    返回对应的二值掩码图像。
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"标签文件不存在: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue  # 无法解析的行，跳过
        cls_id = int(float(parts[0]))
        if keep_classes is not None and cls_id not in keep_classes:
            continue
        coords = list(map(float, parts[1:]))
        # YOLO 检测框格式
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1 = int(round((cx - bw/2) * W)); y1 = int(round((cy - bh/2) * H))
            x2 = int(round((cx + bw/2) * W)); y2 = int(round((cy + bh/2) * H))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W-1, x2), min(H-1, y2)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
            continue
        # YOLO 多边形分割格式
        if len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.array(coords, dtype=np.float64).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * W
            pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * H
            pts_int = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts_int], 255)
    return mask

def mask_dilate(mask: np.ndarray, dilate_px=6):
    """扩大掩码的白色区域 dilate_px 像素"""
    if dilate_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px + 1, 2*dilate_px + 1))
    return cv2.dilate(mask, k, iterations=1)

def mask_core_by_distance(mask: np.ndarray, dist_px=3):
    """
    计算核心区域掩码: 仅保留距离边界至少 dist_px 像素的前景区域。
    """
    if dist_px <= 0:
        return mask.copy()
    bin_img = (mask > 0).astype(np.uint8)
    if bin_img.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 3)
    core = (dist >= dist_px).astype(np.uint8) * 255
    return core

def auto_fix_unit_to_meter(pts_xyz: np.ndarray) -> np.ndarray:
    """
    检查并统一点云单位: 若点云坐标中位深度大于20则视为毫米单位，转换为米。
    """
    pts = pts_xyz.astype(np.float64, copy=False)
    z = pts[:, 2]; z = z[np.isfinite(z)]
    if z.size == 0:
        return pts
    if np.median(z) > 20.0:
        pts = pts / 1000.0
    return pts

def project_points_to_pixels(pts_xyz: np.ndarray, W: int, H: int,
                              fx: float, fy: float, cx: float, cy: float,
                              R_dc: np.ndarray, t_dc: np.ndarray,
                              use_extrinsic=True):
    """
    将3D点投影到图像像素坐标系。
    返回:
      idx_valid: 投影到图像内的点索引集合（对应原始点云列表索引）
      ui, vi: 每个投影点对应的图像像素坐标
      Zc: 相机坐标系下的深度值（用于后续筛选）
    """
    pts = auto_fix_unit_to_meter(np.asarray(pts_xyz))
    valid_mask = np.isfinite(pts).all(axis=1)
    pts = pts[valid_mask]
    base_idx = np.where(valid_mask)[0]
    if pts.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    # 根据标志决定是否将Depth坐标转换到Color坐标系
    pts_c = (pts @ R_dc.T) + t_dc if use_extrinsic else pts
    X, Y, Z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
    front_mask = Z > 1e-6  # 深度为正
    X, Y, Z = X[front_mask], Y[front_mask], Z[front_mask]
    idx_front = base_idx[front_mask]
    # 投影到像素坐标系
    u = fx * (X / Z) + cx; v = fy * (Y / Z) + cy
    ui = np.round(u).astype(np.int32); vi = np.round(v).astype(np.int32)
    inside_mask = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui = ui[inside_mask]; vi = vi[inside_mask]; Z = Z[inside_mask]
    idx_valid = idx_front[inside_mask]
    return idx_valid, ui, vi, Z

def choose_extrinsic_mode(pts_xyz: np.ndarray, core_mask: np.ndarray,
                          W: int, H: int, fx: float, fy: float, cx: float, cy: float,
                          R_dc: np.ndarray, t_dc: np.ndarray):
    """
    判断点云坐标系模式:
    比较点云投影覆盖核心掩码的程度，决定是否应用depth到color的外参。
    返回 (use_extrinsic, hit_depth, hit_color):
      use_extrinsic=True 表示点云需从Depth转换到Color坐标系
      hit_depth/hit_color 为在核心掩码下命中的点数量（供参考）
    """
    idx1, u1, v1, _ = project_points_to_pixels(pts_xyz, W, H, fx, fy, cx, cy, R_dc, t_dc, use_extrinsic=True)
    hit_depth = int((core_mask[v1, u1] > 0).sum()) if idx1.size else 0
    idx2, u2, v2, _ = project_points_to_pixels(pts_xyz, W, H, fx, fy, cx, cy, R_dc, t_dc, use_extrinsic=False)
    hit_color = int((core_mask[v2, u2] > 0).sum()) if idx2.size else 0
    use_extr = (hit_depth >= hit_color)
    return use_extr, hit_depth, hit_color

def adaptive_segment_by_mask_and_3d(pts_xyz: np.ndarray, pts_rgb01: np.ndarray,
                                    mask: np.ndarray, core_mask: np.ndarray,
                                    W: int, H: int, fx: float, fy: float, cx: float, cy: float,
                                    R_dc: np.ndarray, t_dc: np.ndarray, use_extrinsic=True,
                                    dilate_px=4, dbscan_eps=0.06, dbscan_min_points=200,
                                    sor_nb=30, sor_std=2.0, keep_largest_cluster=True):
    """
    执行掩码与点云结合的自适应分割:
    返回: sel (np.bool数组, 表示选择的目标点云索引), refined_mask (uint8图像掩码)
    """
    # 1) 扩大掩码得到候选区域
    mask_big = mask_dilate(mask, dilate_px=dilate_px)
    # 2) 将所有点投影到图像
    idx_valid, ui, vi, Z = project_points_to_pixels(pts_xyz, W, H, fx, fy, cx, cy, R_dc, t_dc, use_extrinsic=use_extrinsic)
    if idx_valid.size == 0:
        raise RuntimeError("没有点投影到图像内，请检查相机参数或点云坐标系！")
    # 3) 选取核心掩码区域内的投影点作为种子点
    seed_flags = (core_mask[vi, ui] > 0)
    seed_indices = idx_valid[seed_flags]
    if seed_indices.size == 0:
        seed_flags = (mask[vi, ui] > 0)
        seed_indices = idx_valid[seed_flags]
    if seed_indices.size == 0:
        raise RuntimeError("没有找到任何种子点，初始mask可能不准确！")
    # 4) 提取候选点：投影在扩张掩码区域内的所有点
    cand_flags = (mask_big[vi, ui] > 0)
    cand_indices = np.unique(idx_valid[cand_flags])
    if cand_indices.size < dbscan_min_points:
        raise RuntimeError(f"候选点太少（{cand_indices.size}），无法执行有效聚类！")
    # 5) 对候选点进行DBSCAN聚类
    cand_pts = auto_fix_unit_to_meter(np.asarray(pts_xyz)[cand_indices])
    cand_pcd = o3d.geometry.PointCloud()
    cand_pcd.points = o3d.utility.Vector3dVector(cand_pts)
    if pts_rgb01 is not None and len(pts_rgb01) == len(pts_xyz):
        cand_cols = np.asarray(pts_rgb01)[cand_indices]
        cand_pcd.colors = o3d.utility.Vector3dVector(cand_cols)
    labels = np.array(cand_pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        target_indices = cand_indices  # 聚类未找到有效簇，则使用全部候选点
    else:
        # 6) 选取含种子点最多的聚类簇
        seed_set = set(seed_indices.tolist())
        best_label = None; best_score = -1; best_size = -1
        for lab in range(labels.max() + 1):
            members_local = np.where(labels == lab)[0]
            members_global = cand_indices[members_local]
            score = sum((idx in seed_set) for idx in members_global.tolist())
            size = members_global.size
            if (score > best_score) or (score == best_score and size > best_size):
                best_score = score; best_size = size; best_label = lab
        if best_label is None or best_score <= 0:
            # 若所有簇都没有种子点，则选择最大簇
            valid_labels = labels[labels >= 0]
            if valid_labels.size == 0:
                raise RuntimeError("DBSCAN未能找到有效的聚类结果！")
            counts = np.bincount(valid_labels)
            best_label = int(np.argmax(counts))
        members_local = np.where(labels == best_label)[0]
        target_indices = cand_indices[members_local]
    # 7) 对目标点云执行统计滤波，去除离群点；可选保留最大连通簇
    target_pts = auto_fix_unit_to_meter(np.asarray(pts_xyz)[target_indices])
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)
    if pts_rgb01 is not None and len(pts_rgb01) == len(pts_xyz):
        target_cols = np.asarray(pts_rgb01)[target_indices]
        target_pcd.colors = o3d.utility.Vector3dVector(target_cols)
    if sor_nb > 0 and sor_std > 0:
        target_pcd, inlier_idx = target_pcd.remove_statistical_outlier(nb_neighbors=int(sor_nb), std_ratio=float(sor_std))
        target_indices = target_indices[np.asarray(inlier_idx, dtype=np.int64)]
    if keep_largest_cluster and len(target_pcd.points) > 0:
        labels2 = np.array(target_pcd.cluster_dbscan(eps=dbscan_eps, min_points=max(20, dbscan_min_points//2), print_progress=False))
        if labels2.size > 0 and labels2.max() >= 0:
            counts2 = np.bincount(labels2[labels2 >= 0])
            main_label = int(np.argmax(counts2))
            main_local_idx = np.where(labels2 == main_label)[0]
            target_pcd = target_pcd.select_by_index(main_local_idx)
            target_indices = target_indices[main_local_idx]
    # 8) 生成布尔选择数组 sel
    sel = np.zeros(len(pts_xyz), dtype=bool)
    sel[target_indices] = True
    # 9) 将目标点投影回2D得到精炼掩码
    idx_v2, u2, v2, _ = project_points_to_pixels(np.asarray(pts_xyz)[sel], W, H, fx, fy, cx, cy, R_dc, t_dc, use_extrinsic=use_extrinsic)
    refined_mask = np.zeros((H, W), dtype=np.uint8)
    if u2.size:
        refined_mask[v2, u2] = 255
        # 闭运算填充小洞并平滑边缘
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, k, iterations=1)
    return sel, refined_mask

# 加载相机内参/外参
Wc, Hc, fx, fy, cx, cy, R_dc, t_dc = load_d435i_params(cam_json_path)

# 确保输出主目录存在
os.makedirs(output_base_dir, exist_ok=True)

# 打开日志文件（写入模式）
with open(log_path, "w", encoding="utf-8") as log_file:
    # 遍历输入目录的所有子目录
    for root, dirs, files in os.walk(input_base_dir):
        # 判断是否为二级目录（包含RGB和PLY子文件夹）
        if 'RGB' in dirs and 'PLY' in dirs:
            rel_path = os.path.relpath(root, input_base_dir)
            out_dir = os.path.join(output_base_dir, rel_path)
            os.makedirs(os.path.join(out_dir, "PLY"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "RGB"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "RGBnew"), exist_ok=True)
            rgb_in_dir = os.path.join(root, "RGB")
            ply_in_dir = os.path.join(root, "PLY")
            # 遍历RGB文件夹下的所有图像文件
            for fname in os.listdir(rgb_in_dir):
                f_lower = fname.lower()
                if not (f_lower.endswith(".jpg") or f_lower.endswith(".jpeg") or f_lower.endswith(".png")):
                    continue  # 跳过非图像文件
                img_path = os.path.join(rgb_in_dir, fname)
                label_path = os.path.splitext(img_path)[0] + ".txt"
                base_name = os.path.splitext(fname)[0]      # 例如 "1234_color"
                ply_base = base_name[:-6] if base_name.endswith("_color") else base_name
                ply_path = os.path.join(ply_in_dir, ply_base + ".ply")
                # 检查标签和点云文件是否存在
                if not os.path.exists(label_path):
                    log_file.write(f"未找到标签文件: {label_path}\n")
                    continue
                if not os.path.exists(ply_path):
                    log_file.write(f"未找到点云文件: {ply_path}\n")
                    continue
                # 读取图像
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    log_file.write(f"图像读取失败: {img_path}\n")
                    continue
                H_img, W_img = img.shape[:2]
                if (W_img != Wc) or (H_img != Hc):
                    print(f"[警告] 图像分辨率 {W_img}x{H_img} 与相机内参 {Wc}x{Hc} 不一致，结果可能有偏差。")
                # 解析YOLO标签为初始mask
                try:
                    mask = parse_yolo_txt_to_mask(label_path, W=W_img, H=H_img)
                except Exception as e:
                    log_file.write(f"解析标签失败: {label_path} ({e})\n")
                    continue
                core_mask = mask_core_by_distance(mask, dist_px=4)
                # 加载点云
                pcd = o3d.io.read_point_cloud(ply_path)
                pts = np.asarray(pcd.points)
                cols = np.asarray(pcd.colors) if pcd.has_colors() else None
                if pts.size == 0:
                    log_file.write(f"点云为空或无法读取: {ply_path}\n")
                    continue
                # 判断点云坐标系模式（是否应用外参）
                use_extrinsic, hit1, hit2 = choose_extrinsic_mode(pts, core_mask, W_img, H_img, fx, fy, cx, cy, R_dc, t_dc)
                # 执行自适应分割提取
                try:
                    sel, refined_mask = adaptive_segment_by_mask_and_3d(
                        pts_xyz=pts,
                        pts_rgb01=cols,
                        mask=mask,
                        core_mask=core_mask,
                        W=W_img, H=H_img,
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        R_dc=R_dc, t_dc=t_dc,
                        use_extrinsic=use_extrinsic,
                        dilate_px=4,
                        dbscan_eps=0.06,
                        dbscan_min_points=200,
                        sor_nb=30,
                        sor_std=2.0,
                        keep_largest_cluster=True
                    )
                except Exception as e:
                    log_file.write(f"处理失败: 图像 {img_path} - {e}\n")
                    continue
                # 保存提取的目标点云
                out_ply_path = os.path.join(out_dir, "PLY", ply_base + ".ply")
                out_pcd = o3d.geometry.PointCloud()
                out_pcd.points = o3d.utility.Vector3dVector(pts[sel])
                if cols is not None and len(cols) == len(pts):
                    out_pcd.colors = o3d.utility.Vector3dVector(cols[sel])
                o3d.io.write_point_cloud(out_ply_path, out_pcd, write_ascii=False, compressed=False)
                # 保存初始掩码作用后的图像（背景置黑）
                cut_img = img.copy()
                cut_img[mask == 0] = 0
                out_mask_img = os.path.join(out_dir, "RGB", fname)
                cv2.imwrite(out_mask_img, cut_img)
                # 保存自适应分割后的抠图图像
                cut_adaptive = img.copy()
                cut_adaptive[refined_mask == 0] = 0
                out_adapt_img = os.path.join(out_dir, "RGBnew", fname)
                cv2.imwrite(out_adapt_img, cut_adaptive)
print("批处理完成！")
