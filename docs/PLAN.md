# 车联网车载感知：YOLO + ByteTrack 多目标跟踪与轨迹生成 —— 实施计划

> 目标：在 BDD100K（车辆/行人/骑行者多类别）与 MOT17（交通子集，行人密集）上，构建
> 「检测 + 跟踪」流水线，实现目标 ID 维持与轨迹生成；横向对比多种跟踪策略，量化
> **ID 切换率 / 跟踪速度 / 轨迹质量**，最终输出可供车联网状态感知使用的结构化轨迹。

---

## 0. 环境与关键约束（已探查）

| 项 | 现状 | 影响与对策 |
|---|---|---|
| GPU | RTX 3050 Laptop, **4 GB** 显存 | 只能用 YOLOv8n/s（640~960）；batch≤8；FP16；MOT17 密集帧用 batch=1 流式 |
| CPU / RAM | 16 核 / **7 GB** | MOT17 帧序列大，**不整解压**，按需 seek 解码；检测结果落盘缓存复用 |
| 磁盘 | 928 GB 可用 | 充足，可全量存 dets/tracks/可视化 |
| Python | `py310` conda env 存在但 torch 为 **CPU 版** | 需装 `torch cu121/cu124` + `ultralytics` + `lap` + `cython_bbox` + `trackeval` |
| 数据集 | 未下载 | 需申请/下载 BDD100K MOT 标签 + MOT17 train |

**核心实验设计原则**：把「检测」与「关联」解耦——先一次性缓存检测结果（jsonl），
所有跟踪策略都在**同一份检测**上跑，这样 ID 切换差异只来自关联逻辑，速度对比也干净。

---

## 1. 阶段总览

| 阶段 | 内容 | 产出 | 预计 |
|---|---|---|---|
| P0 | 环境搭建、数据下载与格式统一 | 可运行环境、`data/converted` | 2~3 d |
| P1 | 检测基线（YOLO）与检测质量评估 | mAP/速度表、缓存检测 | 3 d |
| P2 | 跟踪基线（ByteTrack）跑通 + TrackEval 评测 | 基线指标、轨迹文件 | 3 d |
| P3 | 跟踪策略对比（6~8 组变体）+ 速度基准 | 对比表（IDSW/IDF1/FPS） | 5 d |
| P4 | ID 维持增强（外观/CMC/类别一致性/轨迹补全） | 最优配置 + 消融 | 4 d |
| P5 | 轨迹生成、坐标转换与车联网消息输出 | trajectory 模块 + 消息 schema | 3 d |
| P6 | 可视化、报告、结论 | 图表 + 技术报告 | 3 d |

---

## 2. P0 环境与数据（2~3 d）

### 2.1 环境
```bash
conda create -n v2xmot python=3.10 -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics lap cython_bbox opencv-python pandas scipy motmetrics pyyaml tqdm
# TrackEval（官方 MOT 指标，HOTA/MOTA/IDF1/IDSW）
git clone https://github.com/JonathonLuiten/TrackEval.git && pip install -e TrackEval
```
- 显存保障：`half=True, imgsz=960, batch≤8`；OOM 时降 `imgsz=640`。
- 固定随机种子、记录 `pip freeze` 快照，保证可复现。

### 2.2 数据
**BDD100K（车载视角，多类别，主力数据集）**
- 下载：`bdd100k_mot_labels`（跟踪标签，MOT 格式，含 car/pedestrian/rider/truck/bus/train/motor/bike）
  与对应视频（10k 视频中 20 条标注序列），检测任务可用 `bdd100k_labels_images_*`（10 万图）做 mAP。
- ⚠️ 待核实：MOT 标签采样帧率（5 fps 还是逐帧）与 `seqinfo` 帧率须在 P0 用脚本确认，写入 `data/meta/README`。
- 划分：train 序列调参 → val 序列最终评测（严格不交叉）。

**MOT17（行人密集，遮挡/IDSW 压力测试）**
- train 7 序列（含 3 种公开检测器 det.txt，我们**不引用**公开检测，统一用 YOLO 检测以保证跨数据集公平）。
- 「交通子集」定义：取机动车/人流密集场景 **MOT17-02/04/05/09/10/11/13**，评估时用 MOT17 官方 `ignore/classes/pedestrian` 规则。
- 存储：只解压需要的帧或用 `decord/opencv` 顺序解码；同时保留 `gt/gt.txt`、`seqinfo.ini`。

### 2.3 格式统一（关键工程量）
`src/datasets/convert_*.py` → 统一内部表示：
```
data/converted/<dataset>/<seq>/
  frames/            # 或 frames.txt（索引，不落地图片）
  gt.txt             # MOT 格式：frame,id,x,y,w,h,conf,cls,vis
  seqinfo.ini
```
- BDD 类别映射到统一标签集：`pedestrian, rider, car, truck, bus, motorcycle, bicycle`。
- 评测子集按需过滤（如仅 car+pedestrian 的"车辆/行人"双类主任务，其余类做扩展）。

**交付验收**：`python -m src.datasets.check_data` 打印每序列帧数/框数/类别分布/帧率/忽略区域，无报错。

---

## 3. P1 检测基线（3 d）

1. 模型：`yolov8n.pt`、`yolov8s.pt`（可选 `yolo11n`）COCO 预训练；COCO→统一标签映射（person→pedestrian，motor→motorcycle）。
2. 可选微调：BDD100K 检测训练集上微调 n/s（4 GB 下 `imgsz=640, batch=8`，~10 epoch），对比 zero-shot vs 微调。
3. 检测评估：
   - BDD100K val：mAP50/mAP50-95、per-class AP、小目标(面积<32²)AP。
   - 分辨率消融：640 / 960 / 1280（显存与召回权衡，行人小目标明显受益）。
   - 速度：`preprocess/inference/NMS` 分段计时，FP32 vs FP16。
4. **检测落盘**：对每个评测序列，用最终选定检测配置生成
   `outputs/dets/<det_cfg>/<dataset>/<seq>.jsonl`（frame, cls, score, xyxy），
   供所有跟踪实验复用（可加 `--score-thresh 0.1` 存低分框，兼容 BYTE 二阶段关联）。

**交付**：检测指标表 + 检测速度表 + 缓存检测文件。

---

## 4. P2 跟踪基线 ByteTrack（3 d）

- 实现/封装 `src/track/bytetrack.py`（参考 YOLOX ByteTrack 官方实现，参数外置到 yaml）：
  - 两阶段关联：高分框先与 tracklet 池 IoU 匹配，再用低分框(0.1~0.5)二次匹配；
  - Kalman（SORT 恒速模型）+ 匈牙利匹配 + `track_buffer` 失效保留；
  - 状态机：`New/Tracked/Lost/Removed`，`min_hits`、`max_time_lost`。
- 输出标准 MOT 结果 `outputs/tracks/<exp>/<seq>.txt`（frame,id,x,y,w,h,conf,cls,-1）。
- 评测：TrackEval（MOT17 与 BDD100K 各自配置），输出
  HOTA / DetA / AssA / MOTA / MOTP / IDF1 / IDP / IDR / **IDSW** / Frag / MT / ML / FP / FN。
- 同时用 `py-motmetrics` 交叉校验（注意其 IDF1 定义差异）。

**交付**：ByteTrack 基线指标表，MOT17 与 BDD100K 各一份。

---

## 5. P3 跟踪策略对比（5 d）——核心实验

### 5.1 对比策略矩阵（全部在**同一检测缓存**上跑）
| 编号 | 策略 | 说明 | 关注点 |
|---|---|---|---|
| S0 | Det-Only（每帧框序号当 ID） | 下界参考 | IDSW 爆炸量级 |
| S1 | SORT（IoU + Kalman） | 传统基线 | 遮挡下 ID 断裂 |
| S2 | **ByteTrack 默认** | 两阶段 IoU 关联 | 主基线 |
| S3 | ByteTrack + 参数搜索 | `track_thresh/new_track_thresh/match_thresh/track_buffer` 网格 | 调参收益 |
| S4 | ByteTrack + CMC（GMC/ECC 全局运动补偿） | 车载**运动相机**必做 | 自车运动导致的 IDSW |
| S5 | ByteTrack + ReID 外观（BoT-SORT 风格，OSNet-x0.25） | 轻量 ReID 适配 4 GB | 长遮挡后重连 |
| S6 | ByteTrack + CMC + ReID（BoT-SORT 近似） | 组合上限 | 综合最优 |
| S7 | OC-SORT（观测中心，可选） | 非线性运动/漏检恢复 | 与 S5 对比代价 |

### 5.2 公平性控制
- 检测固定、类别固定、评测帧集固定；只改关联参数。
- 记录每组的 `realtime` 开关状态（ReID 关闭以评速度，开启以评精度）。
- 三档场景难度分组（按 GT 密度/遮挡比例分桶）报告，避免单一均值掩盖差异。

### 5.3 速度基准（`src/eval/speed_bench.py`）
- **关联器速度**：仅跑跟踪器（预加载缓存检测），报告 FPS、每帧关联耗时 P50/P95（ms）、每帧目标数-耗时曲线（1/10/50/100 目标）。
- **端到端速度**：解码 + 检测 + 关联 + 后处理，报告 FPS 与各段占比（bar/pie 图）。
- **显存峰值**：`torch.cuda.max_memory_allocated`。
- 与车联网时延预算对齐：**10 Hz 感知 < 100 ms/帧**是否满足，给出各配置余量。

### 5.4 ID 指标（自研 `src/eval/metrics_id.py`，与 TrackEval 互校）
- **IDSW**（绝对数）与 **ID 切换率** `IDSW / 总匹配数`；
- **ID 维持率** `1 - IDSW/总匹配数`（主 KPI，体现"持续跟踪"能力）；
- **每 GT 轨迹平均 ID 数**（`Σ IDs per GT track / #GT tracks`，理想=1）；
- **IDF1/IDP/IDR**、**Frag（轨迹碎片化次数）**、**MT/ML 比例**；
- **轨迹长度分布**：`≥N 帧的轨迹占比`（N=30/60/120），衡量长时维持；
- 可选 **IDSW 归因分析**：按类别 / 遮挡比 / 目标尺寸 / 交叉场景统计（回答"为什么切换"）。

**交付**：`outputs/eval/summary.csv` 主对比表（行=策略，列=全部指标+速度）+ 4~6 张对比图。

---

## 6. P4 ID 维持增强（4 d）

在 S6 基础上做定向改进与消融：
1. **类别一致性约束**：匹配代价里加入类别一致性（禁止 car↔truck、pedestrian↔rider 误接），BDD 多类别场景收益明显。
2. **轨迹补全/插值**：对 `Lost` 区间做线性或三次样条插值，输出连续轨迹（同时报告插值前后 IDSW/Frag 变化）。
3. **外观库 EMA**：`feat = α·feat_old + (1-α)·feat_new`，`α` 消融；触发 ReID 的条件（`lost_time > τ` 才用外观，控制耗时）。
4. **长间隔 ID 重连（离线）**：全局外观库 + 时空门限，做跨 `track_buffer` 的 ID 合并（离线后处理，报告精度增益与额外耗时，标注"非实时"）。
5. **检测侧协同**：低分框阈值与 `new_track_thresh` 联合调优（提升行人密集场景召回）。

**交付**：消融表（每组 -0/+1 增益）+ 最终推荐配置 `configs/final.yaml`。

---

## 7. P5 轨迹生成与车联网输出（3 d）

### 7.1 轨迹数据结构（`src/traj/trajectory.py`）
逐帧记录 + 逐轨迹汇总：
```json
{"frame_id": 132, "ts": 4.400, "track_id": 17, "cls": "car", "conf": 0.87,
 "bbox": [x1,y1,x2,y2], "center_px": [cx,cy], "velocity_px": [vx,vy],
 "speed_mps": 8.3, "heading_deg": 91.2, "age": 45, "hits": 44, "state": "tracked"}
```
轨迹汇总：`start/end frame`、`持续时间`、`轨迹长度`、`平均/最大速度`、`是否被插值补全`、`ID 切换次数`。

### 7.2 坐标与世界系（车联网关键）
- **路侧固定相机**：提供 `src/traj/homography.py`，四点标定 → 像素到地面平面米制坐标（BEV）；
- **车载移动相机（BDD100K）**：无自车位姿时，输出像素坐标 + 归一化坐标 + 以自车为原点的相对位移（用光流/CMC 估计自车运动做近似），并在报告中明确局限；
- 建议补一个**模拟路侧固定视角流**（同一序列做透视/裁切或使用用户自有路侧视频），演示米制轨迹。

### 7.3 输出格式与接口
- 文件：`outputs/traj/<exp>/<seq>.jsonl`（逐帧）+ `<seq>_tracks.json`（逐轨迹）+ `*.csv`（便于分析）。
- 车联网消息：`src/traj/export_v2x.py` 生成**类 CAM/感知目标列表**的 JSON（含 `station_id, track_id, class, pos[m], vel[m/s], acc, heading, size, conf, age, ttl`），
  可选 MQTT 发布（`--mqtt`）或 ROS2 topic 演示，输出样例报文与 schema 文档 `docs/v2x_schema.md`。
- 质量校验：与 GT 对比的轨迹级误差（仅评估用），丢失率/断链率统计。

---

## 8. P6 可视化与报告（3 d）

- 视频叠加：框+ID+类别+轨迹尾迹（`src/viz/overlay.py`，可导出 mp4/gif 用于答辩）。
- 轨迹平面图：BDD100K/路侧 BEV 轨迹图、ID 切换点标注（红叉）。
- 分析图：IDSW vs 目标密度、FPS vs imgsz、精度-速度散点（Pareto 前沿）。
- 报告结构：任务定义 → 数据与协议 → 方法 → 实验（检测/跟踪/ID/速度）→ 结论与工程建议 → 局限与后续（多相机融合、跨设备 ID 关联、在线 ReID）。

---

## 9. 风险与预案

| 风险 | 预案 |
|---|---|
| 4 GB 显存 OOM（MOT17 密集 + ReID） | imgsz 640、batch 1、ReID 按需（仅 Lost 轨迹）、或改用 MobileNet/OSNet-x0.25 CPU+GPU 混合 |
| 7 GB RAM 不足（大帧缓存） | 不整解压、流式解码、检测缓存 jsonl 压缩、`cv2.setNumThreads(4)` |
| BDD 标签帧率/类别与 COCO 不一致 | P0 脚本核实 + 显式映射表 + 人工抽检 3 段视频 |
| MOT17 只有行人，无车辆 GT | 车辆结论以 BDD100K 为主；MOT17 只做行人压力测试，报告中说明 |
| 车载运动相机导致 IoU 关联失效 | CMC(GMC/ECC) 必做；同时用路侧固定视角做对照实验 |
| 指标工具版本差异 | TrackEval 为准，motmetrics 交叉校验，差异>5% 时排查 GT 过滤规则 |

---

## 10. 目录结构（拟）

```
v2x-mot/
├── configs/        det/*.yaml  track/*.yaml  final.yaml
├── data/           raw/  converted/  meta/
├── src/
│   ├── datasets/   convert_bdd.py convert_mot17.py check_data.py
│   ├── detect/     yolo_detector.py  cache_dets.py
│   ├── track/      base.py sort.py bytetrack.py ocsort.py cmc.py reid.py
│   ├── eval/       run_trackeval.py metrics_id.py speed_bench.py
│   ├── traj/       trajectory.py homography.py export_v2x.py smooth.py
│   └── viz/        overlay.py plot_traj.py plots.py
├── scripts/        00_env.sh 01_data.sh 02_det.sh 03_track.sh 04_eval.sh 05_report.sh
├── outputs/        dets/ tracks/ eval/ traj/ fig/ videos/
└── docs/           PLAN.md  v2x_schema.md  report.md
```

---

## 11. 需要你确认的 5 个决策点

1. **时间预算**：上面按约 3 周紧凑排（可选 6 周含报告打磨），你的截止时间？
2. **主任务类别**：只做 `car + pedestrian` 双类，还是 BDD 全 7 类？
3. **是否微调检测**：仅用 COCO 预训练（快，够用）还是额外在 BDD 微调（+1~2 d，小目标更好）？
4. **"车载"还是"路侧"为主**：BDD100K 是车载运动相机；若你的车联网场景偏 RSU 路侧固定相机，规划里会以固定视角+米制标定为主线，并额外准备一段路侧数据。
5. **是否需要实时消息接口**（MQTT/ROS2）演示，还是只要离线轨迹文件 + schema 文档？
```
```
