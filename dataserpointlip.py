from torch.utils.data import DataLoader
from datasets.plyae_dataset import PLYAutoEncoderDataset
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

train_set = PLYAutoEncoderDataset(
    index_jsonl="indices/train.jsonl",
    root_dir="E:\data\zhj\data_zhj",
    n_points=2048,
    sample_mode="fps",
    normalize=True,
    return_rgb=True,
    rgb_to_tensor=True,
)

loader = DataLoader(train_set, batch_size=8, shuffle=False, num_workers=0, drop_last=True)

batch = next(iter(loader))
pp = batch["points"][0]
print(batch["points"].shape)  # (B, 2048, 3)
df = pd.DataFrame(pp.detach().cpu().numpy(),
                  columns=[f"c{i}" for i in range(pp.shape[1])])
print(batch["label"])  # (B,)
print(batch["meta"]["ply_path"][0])

# 3) 取一个 batch
batch = next(iter(loader))

# 4) 打印 points[0]
print("points shape:", batch["points"].shape)  # (B, 2048, 3)
print(batch["points"][0])  # ✅ 你要的：会打印 2048 行*3列，很长

# （建议额外打印一下前20个点，便于快速看）
print("\nfirst 20 points:\n", batch["points"][0][:20])

# 5) 可视化（3D散点）
pts = batch["points"][0].detach().cpu().numpy()  # (2048, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)  # s=1 点小一点更清楚

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Point Cloud (2048 pts, no normalize)")

# 让坐标轴比例接近一致（不然3D图会被拉扁）
max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
mid = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)

plt.show()
