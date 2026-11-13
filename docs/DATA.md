# 数据集清单与下载计划（BDD100K / MOT17）

> 结论先行：整个课题只需要 BDD100K 的 **Tracking 子集**（`box_track_20` 标签 + `images20-track` 帧），
> 其余 7 个发布件（100k 图像、10k 视频、语义分割、MOTS、车道线、可行驶区域、Pose）**全部不需要**。
> 总量 ≈ 7 GB（含 MOT17），不是 60 GB 也不是 2 TB。

---

## 1. BDD100K 发布件与本课题的取舍

| # | 发布件（官方名） | 大小 | 是否需要 | 说明 |
|---|---|---|---|---|
| 1 | `bdd100k_images_100k.zip`（10 万张 key frame） | 5.6 GB | ❌ 不需要 | 单帧静态图，无时序，无法做跟踪；其 GT 只用于静态检测 mAP，我们已有更强来源（见 #3） |
| 2 | `bdd100k_labels_release.zip`（10 万图 det/属性标签） | ~100 MB | 🟡 **可选** | 唯一价值：提供每张图的 `weather / timeofday / scene` 属性，用于"夜间/雨天"分层报告。见 §4 替代方案 |
| 3 | **`box_track_labels_trainval.zip`**（box_track_20 跟踪标签 JSON） | **120 MB** | ✅ **必需** | 1600 序列（train 1400 / val 200）逐帧 8 类框 + **轨迹 ID** + 遮挡/截断/人群标记 |
| 4 | **`images20-track-{train,val,test}.zip`**（跟踪帧） | 全量 48.5 GB | ✅ **必需（仅取选中序列帧）** | 每序列 203 帧 @5 fps、1280×720；**我们只下筛选出的 40 条序列 ≈ 1.5 GB** |
| 5 | `bdd100k_videos_*.zip`（10k 原始视频） | ~2 TB | ❌ 不需要 | 跟踪帧已直接提供，且 BDD 标签本来就是 5 fps，30 fps 视频用不上 |
| 6 | `bdd100k_seg_track_20_images/labels.zip`（MOTS 实例分割跟踪） | ~5 GB | ❌ 不需要 | 分割掩码，本课题只用 bbox |
| 7 | `bdd100k_seg_2018.zip` / `bdd100k_seg.zip`（语义分割 10k） | ~7 GB | ❌ 不需要 | 不涉及像素级任务 |
| 8 | `bdd100k_drivable_area.zip`（可行驶区域） | ~1 GB | ❌ 不需要 | 若做"轨迹裁剪到路面/伪 BEV"可作扩展，本轮不做 |
| 9 | `bdd100k_lane.zip`（车道线） | ~1 GB | ❌ 不需要 | 同上 |
| 10 | `bdd100k_pose_labels_100k.zip`（人体关键点） | ~1 GB | ❌ 不需要 | 行人不做姿态 |

---

## 2. 具体要用到的字段（已实测确认）

`box_track_labels_trainval.zip` → `bdd100k/labels/box_track_20/{train,val}/<videoName>.json`

```jsonc
[{ "name": "034a8bdf-f9e9ad8d-0000001.jpg",   // 帧名 = <videoName>-<7位帧号>.jpg
   "videoName": "034a8bdf-f9e9ad8d",          // 序列名 = 帧目录名
   "frameIndex": 0,                            // 0-based，对应 5 fps 时间轴
   "labels": [
     { "id": "00101140",                      // ★ 轨迹 ID（GT 的 ID 一致性真值）
       "category": "car",                     // ★ 8 类：pedestrian/rider/car/truck/bus/train/motorcycle/bicycle
                                              //   + other person/other vehicle/trailer → 丢弃（写进 mapping）
       "attributes": {"crowd": false,          // ★ crowd：评测时按忽略区/人群框处理
                      "occluded": true,        // ★ occluded：IDSW 归因分析的关键维度
                      "truncated": false},     // ★ truncated：小目标/边界目标分桶
       "box2d": {"x1":155.5,"y1":236.2,"x2":228.2,"y2":271.1}   // ★ 像素坐标，1280×720
     } ] } ]
```

帧：`http://128.32.162.150/bdd100k/images/track/<split>/<videoName>/<videoName>-<7位帧号>.jpg`
（203 帧 ≈ 40 s @5 fps；实测 1280×720，约 188 KB/帧）

**用途映射**
- `id` → ID 维持/IDSW 评测的唯一真值来源（MOT17 只有行人，BDD 才有多类别车辆）
- `attributes.occluded/truncated` → IDSW 归因四象限（遮挡×尺寸）、策略分组统计
- `category` → 类别一致性约束消融（禁 car↔truck 误接）
- `frameIndex` → 5 fps 时间轴，速度单位 px/frame → px/s 换算基准

---

## 3. 下载计划（总计 ≈ 7 GB / 约 50 min）

| # | 数据 | 大小 | 源（实测可用） | 速度 | 状态 |
|---|---|---|---|---|---|
| D1 | `box_track_labels_trainval.zip` | 120 MB | HF `mrhwl123/BDD100kMOT` | 8.1 MB/s | ✅ 已落地 |
| D2 | 选中 40 条 BDD 序列的逐帧 jpg（≈8 100 帧） | ~1.5 GB | 裸 IP 目录 `http://128.32.162.150/bdd100k/images/track/...` 24 并发 | 2.9 MB/s | 待执行 |
| D3 | `MOT17.zip`（train 7 + test 7） | 5.86 GB | `https://motchallenge.net/data/MOT17.zip` | 3.5 MB/s | 待执行 |
| D4 | `MOT17Labels.zip` | 10 MB | `https://motchallenge.net/data/MOT17Labels.zip` | — | 待执行 |
| D5 | 备选：`images20-track-val-1.zip` | 5.0 GB | HF `vanthanh/BDD100kMOT` | 4.5 MB/s | 仅兜底 |

**不可用源（已排除）**
- `https://bdd-data.berkeley.edu`：TLS 证书与主机名不匹配 + 需登录注册（`curl:60`），HTTP 版 `/download.html` 502。
- Academic Torrents / ModelScope：未检索到可用 BDD100K 跟踪包。

**系统缺口**：无 `unzip`/`p7zip`/`aria2c` → 用 Python `zipfile` 解压；或 `conda install -c conda-forge p7zip`（免 sudo）。

---

## 4. 唯一的数据缺口与替代方案

`box_track_20` **不含** `weather / timeofday / scene` 属性（实测字段仅 name/labels/videoName/frameIndex）。
若报告需要"夜间、雨天对跟踪的影响"分层：

| 方案 | 代价 | 说明 |
|---|---|---|
| A. 亮度启发式（推荐） | 0 | 用已下载帧计算 mean/percentile 亮度 → 分 day/dusk/night；雨天用图像梯度能量+雨滴高频统计粗判（只做定性） |
| B. HF `xmu-xiaoma666-dataset/detection_bdd100k` | 需申请（gated=manual） | parquet 内含 det 标签，可能带 weather/timeofday |
| C. 官方 `bdd100k_labels_release.zip` | 需注册登录 | 站点当前证书异常，成功率低 |

---

## 5. 序列筛选策略（D2 前置，纯标签计算，无需图像）

`src/datasets/select_seqs.py` 对 1600 条序列打分并分层：
- **规模**：car/truck/bus 框数、pedestrian/rider 框数、每帧平均目标数、unique ID 数
- **难度**：`occluded=true` 占比、`truncated=true` 占比、轨迹长度分布（≥100 帧占比）
- **类别多样性**：出现的类别数量
- 分层抽样 → **调参集 30 条（train）** + **评测集 10 条（val，严格不交叉）**
- 输出 `data/meta/bdd_seq_selection.csv`（含每序列分数、选中原因）供你复核

---

## 6. MOT17 处理要点

- 交通子集：`MOT17-02/04/05/09/10/11/13`（train 全部 7 条；test 无 GT 仅用于定性）
- GT 过滤：`gt.txt` 中 `class=7(static)/8(distractor)/12(reflection)` 需过滤，`conf=0` 框保留但按 MOTChallenge 协议计入 ignore；评测忽略区来自 `seqinfo.ini` 的 `imWidth/imHeight` 与 GT 的 `vis` 字段
- **注意**：MOT17 只有 `pedestrian`，车辆相关结论一律以 BDD100K 为准

---

## 7. 目录结构

```
data/
├── raw/
│   ├── bdd100k/
│   │   ├── box_track_labels_trainval.zip          # D1 ✅
│   │   └── images/track/{train,val}/<seq>/*.jpg   # D2
│   └── mot17/{MOT17.zip, MOT17Labels.zip}         # D3 D4
├── converted/            # gt.txt + seqinfo.ini + frames.txt（统一 MOT 格式）
└── meta/                 # bdd_seq_selection.csv, class_mapping.yaml, md5.txt
```

---

## 8. 视频检测（Detection）的数据选择

### 8.1 必选：与跟踪**共用同一批视频帧**

检测不需要另一套数据集——直接把 §3 的 D2/D3 帧同时作为检测输入与检测评测集：

| 数据集 | 帧数 | 帧率 | 检测 GT 来源 | 检测评测用途 |
|---|---|---|---|---|
| BDD 选中序列（10 条 val 评测 + 30 条 train 调参） | 2025 帧（评测）+ 6000 帧（调参） | 5 fps | `box_track_20` JSON → YOLO/COCO | 多类别 mAP、小目标、遮挡子集 |
| MOT17 train 7 序列 | ~8 900 帧 | 25~30 fps | `gt.txt` → 过滤 class 7/8/12 | 密集行人、连续高帧率视频 |

实测 BDD val 10 条序列（5 车辆密集 + 5 行人密集）= **2 025 帧 / 54 276 框**，规模足够算 mAP；
val 200 条序列平均：200 帧、2 328 框/序列、车辆框 2 002、行人框 303。

### 8.2 可选：微调 YOLO 时用 `bdd100k_images_100k`（5.6 GB）

- 只在"决定微调检测器"时才需要；**不微调则完全不需要**。
- 优势：与跟踪帧同源（都抽自同一批 10k 视频），域一致；夜间/雨天/小目标样本多；70k 训练图。
- ⚠️ **数据泄漏风险**：100k images 每视频只取 1 张关键帧，而跟踪序列就是该视频的前 40 s，
  关键帧≈序列第 1 帧。微调时必须**按 `videoName` 把评测序列整条排除**，不能只按图像划分。
- 4 GB 显存可行性：`yolov8n, imgsz=640, batch=8, AMP` ≈ 2~3 GB，单 epoch（70k）约 25~40 min。

### 8.3 可选：路侧固定相机视频集（贴合车联网 RSU 场景）

BDD100K 是**车载运动相机**。若最终要交付"路侧感知"形态，建议补一套固定视角视频：

| 数据集 | 规模 | 帧率 | 类别 | 适配度 |
|---|---|---|---|---|
| UA-DETRAC | 10 h / 8.2 万车辆框 | 25 fps | car/bus/truck | ★★★ 固定路侧杆载、车辆密集，最贴 RSU |
| CityFlow / AI City Challenge | 多路路口视频 | 10~30 fps | 车辆 | ★★★ 路口多相机、可做跨相机 ID |
| VisDrone-MOT | 无人机视角 | 25~30 fps | 车/人 | ★★ 俯视 BEV 友好，但与车载视角差异大 |

> BDD100K 的 30 fps 原始视频（`linxxx3/bdd100k_videos`，19 卷 × 107 GB）实际不可获取，
> 所以"高帧率 + 多类别"只能靠 MOT17（仅行人）或上述路侧集补充。

### 8.4 明确不用于检测评测

- MOT17 **test**（无 GT，仅定性可视化）
- BDD **test** 400 条（无 GT，仅定性 + 无参考帧渲染）
- BDD 100k val 中与评测序列 `videoName` 重合的图（防泄漏）

### 8.5 检测评测协议（与静态图检测不同的部分）

1. **精度**：mAP50 / mAP50-95、per-class AP、小目标 AP（area<32²）、遮挡子集 AP（`occluded=true`）、
   `truncated=true` 子集 AP。
2. **视频特有指标 —— 检测闪烁率（flicker）**：同一目标相邻帧检出状态翻转率
   （借助 GT 轨迹 ID 关联检测结果后统计），反映"框抖动/漏帧"，直接影响 ID 维持率。
3. **BIY 对比**：`640 / 960 / 1280` 输入分辨率的 mAP-显存-FPS 三向权衡（4 GB 显存是硬约束）。
4. **MOT17 检测的 ignore 处理**：`class=7/8/12` 与 `conf=0` 框必须按 ignore 参与评测，
   COCO 的 `iscrowd=1` 是唯一可行近似（或直接用 MOTChallenge 的检测评测脚本）。
5. **速度**：preprocess / inference / NMS 分段计时，FP32 vs FP16，batch=1 流式（在线场景真实值）。

---

## 9. 落盘前校验清单

1. `md5sum -c` 对照 HF 提供的 `.md5`
2. 每序列：帧文件数 == 标签帧数 == `frameIndex` 最大值+1
3. 逐帧：标签框数 == GT 行数；框坐标越界数 = 0
4. 类别直方图与 §2 一致（含被丢弃类别计数）
5. 随机 3 条序列 × 10 帧渲染 GT 叠加图，人工确认无错位（BDD 标签偶有噪声，需记录发现的问题样本）
6. `python -m src.datasets.check_data` 全绿并生成 `data/meta/data_report.md`
