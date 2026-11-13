# v2x-mot · **YOLO11** + ByteTrack/BoT-SORT 多目标跟踪（车联网车载感知）

> **当前版本：`v1.0-yolo11`** —— 检测器固定为 **Ultralytics YOLO11s**（`yolo11s.pt`）。
> 后续若切换到 YOLOv8 / YOLOv12 或 RT-DETR，另起 `v2.x-<detector>` 版本，本版本号不再改动。

![version](https://img.shields.io/badge/version-v1.0--yolo11-blue)
![detector](https://img.shields.io/badge/detector-YOLO11s-00b3a4)
![tracker](https://img.shields.io/badge/trackers-ByteTrack%20%7C%20BoT--SORT-orange)
![dataset](https://img.shields.io/badge/dataset-BDD100K%20Tracking-green)
![python](https://img.shields.io/badge/python-3.10-yellow)

在 BDD100K Tracking 子集上，用 **YOLO11 + ByteTrack / BoT-SORT** 做车辆与行人的持续跟踪，
量化 **ID 维持率 / ID 切换率 / IDF1 / MOTA / 帧率与延迟**，并输出可供车联网状态感知使用的**轨迹与 V2X 消息**。

---

## 亮点

| 项 | 数值 | 说明 |
|---|---|---|
| **ID 维持率** | **96.35%** | 3 条评测序列平均，`1 − IDSW/TP` |
| ID 切换率 | 3.65% | 同上，BoT-SORT + 光流运动补偿 |
| IDF1 / MOTA | 0.496 / 0.397 | 5 种跟踪策略中最优 |
| 端到端帧率 | **40.4 FPS** | imgsz 1280、FP16、单帧流式（ByteTrack） |
| 帧延迟 P95 | 32~38 ms | 对照 10 Hz 车联网感知预算（≤100 ms）**余量 2.6 倍** |
| 数据规模 | 140 序列 / 27 333 帧 | 标签侧覆盖 BDD Tracking 全量 1600 序列 |

---

## 结果速览

### 1. 跟踪策略对比（3 条 eval 序列平均，检测固定）

| 策略 | IDF1 | MOTA | IDSW | ID 切换率 | **ID 维持率** | MT |
|---|---|---|---|---|---|---|
| **BoT-SORT + CMC（光流）** | **0.496** | **0.397** | **131** | **3.65%** | **96.35%** | **33.7** |
| ByteTrack @5fps 重调 | 0.464 | 0.354 | 189 | 6.03% | 93.97% | 21.7 |
| 近似 SORT（关低分框二阶段） | 0.454 | 0.345 | 189 | 6.26% | 93.74% | 19.7 |
| ByteTrack 官方默认 | 0.435 | 0.342 | 215 | 7.68% | 92.32% | 17.7 |
| ByteTrack 过宽松 | 0.425 | 0.322 | 193 | 6.00% | 94.00% | 21.7 |

> 结论：**自车运动是车载场景 ID 断裂的主因**，加光流运动补偿（CMC）即可 IDF1 +3.2 pt、IDSW −31%、MT +55%。

### 2. 检测器：COCO 预训练 vs BDD 微调（3 条 eval 序列，IoU 0.5，imgsz 1280）

| 模型 | conf | 行人 R/P | 轿车 R/P | 合计 R/P |
|---|---|---|---|---|
| **YOLO11s (COCO)** | 0.05 | **55% / 56%** | **72% / 56%** | **69% / 55%** |
| YOLO11s 微调 FT40（40 序列） | 0.05 | 32% / 63% | 68% / 66% | 63% / 66% |
| YOLO11s 微调 FT2（120 序列 + 行人过采样） | 0.05 | 43% / 52% | 69% / 56% | 65% / 56% |

> 结论：小数据微调只提升精确率（+5~10 pt），**行人类别仍以 COCO 预训练最好**；
> 卡车类别两轮微调后召回仍≈0%，根因是 BDD 与 COCO 的 `truck` 语义不一致（标注问题，非训练问题）。

### 3. 场景化表现（4 条序列，BoT-SORT+CMC）

| 场景 | IDF1 | MOTA | IDSW | ID 维持率 |
|---|---|---|---|---|
| 车辆密集（4200 车） | 0.585 | 0.482 | 96 | 96.9% |
| 车辆+卡车 | 0.519 | 0.342 | 81 | 96.6% |
| 行人密集（3779 人） | 0.491 | 0.476 | 272 | 92.4% |
| 行人+骑行者 | 0.517 | 0.366 | 130 | 95.9% |

---

## 快速开始

```bash
# 0) 环境（Python 3.10 + CUDA 12.4）
conda create -n v2xmot python=3.10 -y && conda activate v2xmot
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.4.211 lap polars nvidia-ml-py \
    "opencv-python-headless==4.10.0.84" pandas scipy matplotlib tqdm remotezip

# 1) 数据（下载 → 抽帧 → 划分 40/10/10 → 转 YOLO 格式）
bash reproduce.sh 1 && bash reproduce.sh 2

# 2) 跑一条视频的跟踪（YOLO11 + ByteTrack/BoT-SORT）
python scripts/track_video.py \
  --source data/raw/bdd100k/frames/<序列名> \
  --model yolo11s.pt --fps 5 --imgsz 1280 --conf 0.05 --half \
  --tracker configs/track/botsort_cmc.yaml --tag best
# → out/<序列名>_best.mp4 / .jsonl / _tracks.csv / _stats.json

# 3) 指标 / V2X 消息 / 轨迹图 / 测速
python scripts/eval_tracks.py --jsonl out/<序列名>_best.jsonl --seq <序列名>
python scripts/export_v2x.py  --jsonl out/<序列名>_best.jsonl --fps 5 --assume-scale 1108
python scripts/viz_traj.py    --jsonl out/<序列名>_best.jsonl --seq <序列名>
python scripts/bench_speed.py --source data/raw/bdd100k/frames/<序列名> --imgsz 1280 --tag cmc

# 4) 复现性校验（与 repro/golden.json 对比，期望 21/21 通过）
bash reproduce.sh 5
```

---

## 目录结构

```
├── reproduce.sh            一键复现入口（0~5 / all）
├── RELEASE_NOTES.md        版本说明（v1.0-yolo11）
├── scripts/                15 个 CLI（下载/抽帧/转格式/训练/评测/跟踪/V2X/可视化/测速/复现校验）
├── configs/track/          5 套跟踪策略（ByteTrack ×4 + BoT-SORT + CMC）
├── docs/                   10 份文档（方案/数据规格/代码讲解/结果/V2X schema/gap 分析）
├── repro/                  复现要件：golden.json + env 快照 + 说明
├── data/meta/              划分清单 split.json、序列筛选、分片映射
└── runs/README.md          训练权重说明（权重不入库，见 Release 附件）
```

> 大数据（`data/raw`、`runs/`、`out/*.mp4`）与 `*.pt` 权重**不入库**；
> 需要"代码 + 文档 + 权重 + 结果表"的完整快照，请下载 Release 中的
> `v1.0-yolo11_<date>.tar.gz`（约 34 MB）。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | 全部实验结果与结论 |
| [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) | 14 个脚本的讲解、数据流、踩坑点 |
| [`repro/README.md`](repro/README.md) | 复现等级声明、预算、7 个已知坑 |
| [`docs/DATA_SPEC.md`](docs/DATA_SPEC.md) | BDD100K 各子集的取舍与规格 |
| [`docs/V2X_SCHEMA.md`](docs/V2X_SCHEMA.md) | 车联网状态消息字段定义 |
| [`docs/TRACKER_VS_DATA.md`](docs/TRACKER_VS_DATA.md) | 跟踪策略与数据集特性的依赖关系（含实测证据） |
| [`docs/PLAN.md`](docs/PLAN.md) | 整体技术方案 |
| [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md) | 现状与目标的差距分析 |

---

## 版本演进

| 版本 | 检测器 | 跟踪 | 关键内容 |
|---|---|---|---|
| **v1.0-yolo11**（当前） | **YOLO11s** | ByteTrack ×4 / BoT-SORT+CMC | 5 策略对比、IDSW/IDF1 评测、V2X 输出、可复现打包 |

---

## 已知局限

1. 评测仅 3/10 条序列（另有 7 条已就绪未跑）
2. 行人召回 55%，仍有提升空间
3. 卡车类别基本失效（BDD 标注语义问题）
4. 米制坐标依赖假设焦距（BDD 无相机标定）→ 真实米制轨迹需 KITTI Tracking
5. MOT17 交通子集尚未纳入跟踪实验
6. 检测未缓存，策略对比需重跑检测

---

## 数据来源声明

- 数据集：[BDD100K](https://bdd-data.berkeley.edu/)（Tracking 子集，遵循其 [LICENSE](https://doc.bdd100k.com/download.html)）
- 检测/跟踪框架：[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)、ByteTrack、BoT-SORT
- 本仓库仅包含代码、文档与指标表，**不重新分发原始图像数据**；训练权重见 Release 附件
