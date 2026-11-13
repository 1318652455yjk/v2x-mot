# Release v1.0-yolo11（2026-11-13）

> 车联网车载感知：YOLO11 + ByteTrack/BoT-SORT 多目标跟踪，ID 维持、轨迹生成、策略对比、V2X 状态消息输出。
> 本快照为**可复现的冻结版本**，后续改进请另起版本（v1.1+）。

## 一、这个版本做了什么

| 环节 | 内容 | 关键产出 |
|---|---|---|
| 数据 | BDD100K Tracking 子集：标签全量（1600 序列）+ 帧（140 序列 / 27 333 帧 / 3.95 GB） | `data/meta/split.json` |
| 划分 | train 40 / intval 10 / eval 10，按序列切分、两两无交集 | `data/meta/split.json` |
| 检测 | YOLO11s（COCO 预训练）+ 两轮 BDD 微调（FT40 / FT2） | `weights/bdd_ft2_best.pt` |
| 跟踪 | ByteTrack 4 配置 + BoT-SORT(CMC)，5 种策略对比 | `configs/track/*.yaml` |
| 指标 | IDSW / ID 切换率 / ID 维持率 / IDF1 / MOTA / Frag / MT-ML | `results/*.json` |
| 速度 | 分段耗时（读图/预处理/推理/关联）+ P50/P95 + 显存 | `results/speed_*.json` |
| 输出 | 逐帧轨迹 jsonl、逐轨迹 csv、轨迹图、ID 切换点图、**V2X 状态消息 + schema** | `scripts/export_v2x.py`、`docs/V2X_SCHEMA.md` |

## 二、核心结论（本版本）

### 检测（3 条 eval 序列，607 帧，IoU 0.5，imgsz 1280）

| 模型 | conf | 行人 R/P | 轿车 R/P | 合计 R/P |
|---|---|---|---|---|
| **COCO yolo11s** | 0.05 | **55% / 56%** | 72% / 56% | **69% / 55%** |
| 微调 FT40 | 0.05 | 32% / 63% | 68% / 66% | 63% / 66% |
| 微调 FT2 | 0.05 | 43% / 52% | 69% / 56% | 65% / 56% |

- **COCO 预训练在行人类别上仍最优**；微调只提升精确率（+5~10 pt）
- **卡车召回两轮微调后仍≈0%**：BDD 与 COCO 的 `truck` 语义不一致（标注问题，非训练问题）

### 跟踪策略（3 条 eval 序列平均）

| 策略 | IDF1 | MOTA | IDSW | ID 维持率 | MT |
|---|---|---|---|---|---|
| **BoT-SORT + CMC** | **0.496** | **0.397** | **131** | **96.35%** | **33.7** |
| ByteTrack @5fps 重调 | 0.464 | 0.354 | 189 | 93.97% | 21.7 |
| 近似 SORT | 0.454 | 0.345 | 189 | 93.74% | 19.7 |
| ByteTrack 官方默认 | 0.435 | 0.342 | 215 | 92.32% | 17.7 |
| ByteTrack 过宽松 | 0.425 | 0.322 | 193 | 94.00% | 21.7 |

- **光流运动补偿（CMC）是最大单项收益**：IDF1 +3.2 pt、IDSW −31%、MT +55%
- **单序列结论不可外推**：`b1e9ee0e` 单条上 CMC 反而更差（0.543 vs 0.596）

### 速度（imgsz 1280，单帧流式）

| 配置 | FPS | P50 | P95 | 推理 |
|---|---|---|---|---|
| ByteTrack @5fps | 40.4 | 23.7 ms | 32.3 ms | 13.2 ms |
| BoT-SORT+CMC | 31.9 | 30.5 ms | 38.0 ms | 13.2 ms |

- 对照 10 Hz 车联网感知预算（≤100 ms/帧）：**余量 2.6 倍以上**

### 场景化演示（4 条序列，`out/demo/`）

| 场景 | IDF1 | MOTA | IDSW | ID 维持率 |
|---|---|---|---|---|
| 车辆密集（4200 车） | 0.585 | 0.482 | 96 | 96.9% |
| 车辆+卡车 | 0.519 | 0.342 | 81 | 96.6% |
| 行人密集（3779 人） | 0.491 | 0.476 | 272 | 92.4% |
| 行人+骑行者 | 0.517 | 0.366 | 130 | 95.9% |

## 三、包内容

```
v1.0-yolo11_<date>/
├── scripts/     14 个 CLI（下载/抽帧/转格式/训练/评测/跟踪/V2X/可视化/测速）
├── configs/     5 套跟踪策略 yaml
├── docs/        10 份文档（方案、数据规格、代码讲解、结果、V2X schema、gap 分析…）
├── data/meta/   划分清单、序列筛选、分片映射
├── weights/     bdd_ft2_best.pt（mAP50 0.427）、bdd_ft_best.pt（0.363）
└── results/     策略对比表、逐序列评测、速度基准 json
```

## 四、环境（复现所需）

```bash
conda create -n v2xmot python=3.10 -y
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics lap polars nvidia-ml-py "opencv-python-headless==4.10.0.84" \
            pandas scipy matplotlib tqdm remotezip
```

## 五、已知局限（v1.1 待改进）

1. 评测仅 3 条序列（eval 共 10 条，另有 7 条未跑）
2. 行人召回 55%，仍有提升空间
3. 卡车类别基本失效（标注语义）
4. 米制坐标依赖假设焦距（BDD 无标定）→ 需 KITTI Tracking 才能给可信 m/s、heading
5. MOT17 交通子集尚未纳入
6. 检测未缓存，策略对比需重跑检测（13 ms/帧）
7. 代码未包化（IoU/读标签逻辑重复 3 份）
