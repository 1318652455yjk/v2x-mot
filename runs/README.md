# 训练权重（未入库，仅本地）

| 目录 | 说明 | mAP50 | 权重 |
|---|---|---|---|
| `detect/runs/bdd_ft/` | 微调第 1 轮：40 序列 / 7 885 图 | 0.363 | `weights/best.pt` (19 MB) |
| `detect/runs/bdd_ft2/` | 微调第 2 轮：120 序列 + 行人过采样 / 26 833 图 | **0.427** | `weights/best.pt` (19 MB) |
| `detect/runs/bdd_yolo11s/` | 早期 AdamW 试验（已弃用） | 0.362 | — |
