# 代码讲解（v2x-mot）

> 14 个脚本，按「下载 → 数据准备 → 训练 → 检测评测 → 跟踪 → 跟踪评测 → 输出」7 个环节组织。
> 所有脚本都是独立 CLI，互相通过**文件**（jsonl/csv/json/yaml）耦合，不共享内存状态。

## 0. 全景

```
                    ┌──────────────── 下载层 ────────────────┐
 HF/hf-mirror ──▶ hf_download.py（8 线程分块并行）
                    │  → zips/images20-track-*.zip
 BDD 标签 JSON ──▶ bdd_to_yolo.py ─┐
                    │                ├─▶ data/yolo{,2}/images|labels/*.txt
 帧 zip ──▶ prepare_local.py ──────┘   （图片用软链接，不复制）
              （抽帧 + train/intval/eval 划分 → data/meta/split.json）
                                                   │
                    ┌──────────────── 训练层 ──────▼─────────┐
                    │  train_yolo.py（ultralytics YOLO.train）
                    │  → runs/detect/runs/<name>/weights/best.pt
                    └───────────────────────────────────────┘
                                                   │
        ┌──────────────── 评测层 ──────────────────┼──────────────┐
        │ eval_det.py（单帧 P/R，分类别）           │              │
        └──────────────────────────────────────────┼──────────────┘
                                                   ▼
                    ┌──────────────── 跟踪层 ────────────────┐
                    │ track_video.py  ← configs/track/*.yaml │
                    │   → out/<seq>_<tag>.mp4 / .jsonl /     │
                    │      _tracks.csv / _stats.json         │
                    └────────────────┬───────────────────────┘
                                     │
        ┌──────────────── 输出/评测层 ▼──────────────────────────┐
        │ eval_tracks.py   → IDSW/IDF1/MOTA/Frag（对 GT）        │
        │ export_v2x.py    → V2X 状态消息 jsonl + schema          │
        │ viz_traj.py      → 轨迹图 / ID 切换点图                 │
        │ bench_speed.py   → 分段耗时 + FPS                       │
        └────────────────────────────────────────────────────────┘
```

辅助脚本：`select_seqs.py`（选序列）、`make_video.py`（帧→mp4）、`fetch_bdd_seq.py`/`fetch_batch.py`（早期按需抽帧，现已被本地 zip 方案取代）。

---

## 1. 下载层

### `hf_download.py`（124 行）— 大文件并行下载
```
resolve(name)   走 hf-mirror 的 resolve → 拿到 307 跳转的签名 URL + 文件大小
                （签名 1 小时过期，所以每次重试都重新解析）
grab(url,s,e,p) 下载 [s,e] 字节区间到 .part 文件；发现已有字节则续传
download(...)   按 64 MB 切片 → 8 线程并发 → 按序拼接 → 校验大小
main()          支持多个文件名；已存在则跳过（幂等）
```
**为什么要写它**：单连接只有 1.4 MB/s，8 并发能到 16~68 MB/s；10 GB 从 2 小时降到 18 分钟。

### `prepare_local.py`（129 行）— 从本地 zip 抽帧 + 三份划分
```
seq_stats(label_zip, seq, split)  统计每序列：帧数/车辆框/行人框/每帧密度/类别数
pick(cands, n, stats, key)        按某指标排序取前 n
main()  ① 评测集全部取自 val zip（含 --keep-eval 指定的固定序列）
        ② 训练/内验全部取自 train zip，且与评测集不重叠
        ③ 把选中序列的帧从 zip 解到 data/raw/bdd100k/frames/<seq>/
        ④ 写 data/meta/split.json
```
**关键点**：以**序列**为单位划分，杜绝帧级泄漏（同序列相邻帧高度相似）。

### `bdd_to_yolo.py`（116 行）— 标签格式转换
```
CLASSES = [pedestrian, rider, car, truck, bus, train, motorcycle, bicycle]
DROP    = {other person, other vehicle, trailer}   # 与 COCO 语义不对应，丢弃
每帧：框裁剪到图内 → 丢 w/h<2px 的退化框 → 归一化 cx,cy,w,h 写 .txt
图片：os.symlink 到 data/yolo/images/<split>/<seq>__<frame>.jpg（不复制，省磁盘）
末尾：生成 data.yaml（nc=8 + names）
```

---

## 2. 训练层

### `train_yolo.py`（42 行）
```python
model = YOLO("yolo11s.pt")
model.train(data=..., epochs, imgsz=640, batch, device=0, amp=True,
            optimizer='SGD', lr0=0.005, cos_lr=True, warmup_epochs=1,
            mosaic=0.8, scale=0.4, close_mosaic=8, patience=12, seed=0, plots=False)
```
超参选择理由：4 GB 显存 → batch 8~12 / imgsz 640 / AMP；小数据微调 → SGD + 低 lr0 + 余弦退火；`close_mosaic` 让最后几轮关掉 mosaic 以稳定收敛。

---

## 3. 检测评测层

### `eval_det.py`（155 行）— 分类别 precision/recall
```
iou_matrix(a,b)             向量化 IoU（一次算 N×M）
idx2bdd                     模型类别索引 → BDD 类别名（自动区分 COCO / 微调模型）
主循环：每帧一次 predict（conf 取最小阈值）→ 再在后处理里按多个 conf 过滤
        每个类别做 Hungarian 匹配（IoU≥0.5）→ TP/FP/FN
输出：分类别 召回/精确 表（可多 conf 并排）
```
**设计要点**：一次推理扫多个阈值，避免重复跑模型；只评测 GT 里属于 8 类的框。

---

## 4. 跟踪层（核心）

### `track_video.py`（277 行）
```
frames_iter(source)  统一迭代器：目录→按文件名排序读 jpg；mp4→逐帧 cv2.read
                     （自己控循环，不用 model.track(source=)，为了精确计时）
main()
├─ 载入 model，解析 model.names
│     is_bdd = "pedestrian" in names      # 自动判断微调模型
│     classes 默认：COCO→[0,1,2,3,5,7]；微调→全部
├─ 逐帧：
│    t0 → model.track(img, persist=True, tracker=yaml, conf, imgsz, half)
│    t1 → lat.append(ms)                  # 每帧延迟，用于 P50/P95
│    解析 boxes → xyxy/conf/cls/id
│    算 center、vel_px（与上一帧中心差）、speed_px_s（× fps）
│    写 jsonl 一行一目标
│    更新 tracks[id]（完整历史）和 hist[id]（定长队列，画尾迹）
│    画：尾迹折线 → 框 → 类别#ID 标签 → 左上角 HUD
└─ 收尾：tracks.csv（逐轨迹汇总）+ stats.json（FPS/延迟/ID数/轨迹长度分布）
```
**三个易错点**
1. `persist=True` 才在视频内维持 ID；换视频必须新建模型或调 `predictor.trackers[0].reset()`
2. `track_buffer` 是**帧数**，要按 `期望容忍秒数 × fps` 换算（5 fps 上默认 30 帧 = 6 秒，太久）
3. ultralytics 里 `match_thresh` 是「1 − IoU」代价门限，**越大越宽松**（调小会突然变严，ID 爆炸）

### `configs/track/*.yaml`
| 文件 | 关键差异 | 地位 |
|---|---|---|
| `bt_default.yaml` | 官方默认，buffer 30 | 对照基线 |
| `bt_5fps.yaml` | buffer 12、match 0.92、thresholds 降低 | 5 fps 重调基线 |
| `bt_5fps_loose.yaml` | match 0.97、buffer 6 | 过度放宽（负例） |
| `sort_like.yaml` | `track_low_thresh = track_high_thresh` | 近似 SORT（关掉低分框二阶段） |
| `botsort_cmc.yaml` | `tracker_type: botsort` + `gmc_method: sparseOptFlow` + `with_reid: False` | **最终最优** |

---

## 5. 跟踪评测层

### `eval_tracks.py`（192 行）— MOTChallenge 风格指标
```
load_gt(zip, seq)    读 box_track_20，保留 8 类，类别存"组"，
                     GT2DET 允许多别名：pedestrian → (person, pedestrian)
                     （这样同一份代码能评 COCO 模型和微调模型）
load_pred(jsonl)     按帧分组
逐帧：iou_matrix → 代价矩阵 = -IoU（类别同组时）否则 1e6
      linear_sum_assignment（IoU≥0.5）→ tp / 匹配对 (gt_id, pred_id)
      IDSW：同一 GT 轨迹的匹配预测 ID 发生变化
IDF1：在 (gt_id × pred_id) 计数矩阵上再做一次全局 Hungarian → IDTP → IDF1/IDP/IDR
Frag：GT 轨迹被覆盖帧的断档 + IDSW 次数
MT/ML：覆盖率 ≥0.8 / ≤0.2
输出：JSON（含 ID_keep_rate = 1 − IDSW/TP、IDS_per_gt_track）
```

### `bench_speed.py`（128 行）
```
预热 warmup 帧 → reset tracker → 逐帧计时
分段：读图 / preprocess / inference / postprocess+关联（ultralytics 的 r.speed）
输出 FPS、P50/P95、平均目标数、显存峰值 + 分段条形图 + 帧耗时直方图
```

---

## 6. 输出层

### `export_v2x.py`（144 行）
```
heading_deg(vx,vy)   atan2(vx, -vy) → 0°=图像正上（远方），顺时针为正
每帧组一条消息：{station_id, frame, ts, ttl, n_obj, objects[]}
目标字段：track_id, cls, cls_code, bbox_px, center_px, center_norm,
          size_px, vel_px_s, heading_deg, conf, state
可选 --assume-scale：针孔模型 + 地面平面 → pos_m_est / speed_mps_est（粗略，仅演示）
首次运行自动写 docs/V2X_SCHEMA.md
```

### `viz_traj.py`（143 行）
```
traj_<tag>.png        每条轨迹一色，圆点=起点，方块=终点
traj_stats_<tag>.png 目标数-时间曲线 + 轨迹长度直方图
idsw_<tag>.png       与 GT 比对，在切换位置画红叉（需 --seq）
```

---

## 7. 辅助脚本

| 脚本 | 作用 |
|---|---|
| `select_seqs.py` | 只读标签给 1600 条序列打分（车辆/行人密度、遮挡、轨迹长度、类别多样性）→ 分层抽样名单 |
| `make_video.py` | 图片序列 → mp4；`--dup N` 做平滑播放版；`--gt-seq` 叠 GT 框 |
| `fetch_bdd_seq.py` / `fetch_batch.py` | 早期方案：从 HF zip 用 HTTP Range 只抽指定序列（网络差时用；现已被本地 zip 取代） |

---

## 8. 一个完整实验的命令序列

```bash
Q=~/miniconda3/envs/py310/bin/python

# ① 下载（只需一次）
$Q scripts/hf_download.py images20-track-train-1.zip images20-track-val-1.zip --out data/raw/bdd100k/zips

# ② 抽帧 + 划分
$Q scripts/prepare_local.py --n-train 40 --n-intval 10 --n-eval 10

# ③ 转 YOLO 格式（train / val）
$Q scripts/bdd_to_yolo.py --seqs-file data/meta/yolo_train.csv --split train
$Q scripts/bdd_to_yolo.py --seqs-file data/meta/yolo_val.csv   --split val

# ④ 微调（可选）
$Q scripts/train_yolo.py --name bdd_ft --epochs 30

# ⑤ 检测评测（COCO vs 微调）
$Q scripts/eval_det.py --model yolo11s.pt --group eval --limit-seq 3 --confs 0.05,0.10

# ⑥ 跟踪 + 评测 + 输出
$Q scripts/track_video.py --source data/raw/bdd100k/frames/<seq> \
   --model yolo11s.pt --fps 5 --imgsz 1280 --conf 0.05 --half \
   --tracker configs/track/botsort_cmc.yaml --tag best
$Q scripts/eval_tracks.py --jsonl out/<seq>_best.jsonl --seq <seq>
$Q scripts/export_v2x.py  --jsonl out/<seq>_best.jsonl --fps 5
$Q scripts/viz_traj.py    --jsonl out/<seq>_best.jsonl --seq <seq>
$Q scripts/bench_speed.py --source data/raw/bdd100k/frames/<seq> --imgsz 1280 --tag cmc
```

---

## 9. 代码质量现状与改进建议

| 现状 | 问题 | 建议 |
|---|---|---|
| 14 个独立脚本 | 无包结构，重复的 IoU/读标签代码出现 3 次（`eval_det`/`eval_tracks`/`viz_traj`） | 抽成 `src/common/{iou.py, bdd_labels.py}` |
| 检测未缓存 | 换跟踪参数必须重跑检测（13 ms/帧 → 只为了换关联逻辑不划算） | 加 `cache_dets.py` 落盘 `dets/<cfg>/<seq>.jsonl` |
| 评测是自研版 | 无 HOTA/AssA，不能直接对标论文 | 接 TrackEval |
| 无单元测试 | 参数语义（match_thresh）已经踩过坑 | 给 IoU/匹配/IDSW 加 3~5 个 pytest |
| 硬编码 1280×720 | `bdd_to_yolo` 的裁剪上限、`export_v2x` 的归一化 | 从图片实际尺寸读取 |
