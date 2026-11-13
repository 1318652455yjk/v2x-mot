# 精简方案：单视频 YOLO11 + ByteTrack 跟踪

> 范围收窄：不做多数据集大规模评测，**只在一段视频上跑通 YOLO11 + ByteTrack**，
> 产出：持续跟踪 + ID 维持 + 轨迹输出 + 策略对比（ID 切换率 / 速度）。

---

## 1. 目标与产出

| 产出 | 形式 |
|---|---|
| 跟踪视频 | `out/<seq>_bytetrack.mp4`（框 + ID + 类别 + 轨迹尾迹） |
| 轨迹数据 | `out/<seq>_tracks.jsonl`（逐帧）+ `<seq>_tracks.csv`（逐轨迹汇总） |
| 策略对比表 | `out/compare.csv`（SORT / ByteTrack / ByteTrack+调参 / BoT-SORT × IDSW / IDF1 / FPS） |
| 曲线图 | 轨迹平面图、ID 数演变、FPS 分布 |
| （有 GT 时）指标 | IDSW / IDF1 / MOTA / Frag + FPS（单序列 TrackEval） |

## 2. 单视频选哪条

| 选项 | 视频 | 帧率 | 大小 | GT | 优点 | 代价 |
|---|---|---|---|---|---|---|
| **A（推荐先做）** | BDD100K val 选 1 条车辆密集序列 | 5 fps / 203 帧 / 1280×720 | ~40 MB | ✅ 有 | 车载视角、多类别、**最快能跑** | 5 fps 画面偏卡 |
| B | MOT17 train 选 1 条（如 04/10） | 25~30 fps | 需下 5.86 GB zip | ✅ 有 | 视频流畅，适合演示 | 行人为主，非车载 |
| C | 你自己的行车记录仪 mp4 | 任意 | 自备 | ❌ 无 | 最贴真实场景 | 无 GT → IDSW 只能人工抽检 |

- 有 GT ⇒ 能算**真实 ID 切换率**；无 GT ⇒ 只能给"轨迹数量/长度/中断次数/FPS"+ 人工抽检 IDSW。
- 脚本设计成 `--source` 可切换，A 跑通后换成 C 的视频无需改代码。

## 3. 技术栈与关键参数（4 GB 显存约束）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics lap          # ultralytics>=8.3 支持 YOLO11
```

```python
from ultralytics import YOLO
model = YOLO("yolo11s.pt")                       # n 更快、s 更准；3050 4GB 两者都行
results = model.track(
    source="input.mp4",
    tracker="bytetrack.yaml",      # 或 botsort.yaml（ReID 版）
    classes=[0,1,2,3,5,7],         # person,bicycle,car,motorcycle,bus,truck（COCO 索引）
    conf=0.25, iou=0.5,
    imgsz=960, half=True,          # 4GB 显存下 960+FP16 安全
    persist=True,                  # ★ 视频内保持轨迹状态（必须）
    verbose=False,
)
```

**必须注意的 3 点**
1. `persist=True` 才能在视频内维持 ID；换视频必须新建 tracker（否则 ID 跨视频串号）。
2. `track_buffer` 默认 30 帧 = 6 s@5fps（BDD）/1 s@30fps（MOT17）→ **必须按帧率换算**，否则 BDD 上 ID 保留过久会误接。
3. 4 GB 显存：`imgsz=960, half=True, batch=1`；若 OOM 降到 640 或换 `yolo11n`。

## 4. 策略对比（在同一视频上）

| 编号 | 配置 | 说明 |
|---|---|---|
| T1 | SORT（IoU only） | 下界基线 |
| T2 | **ByteTrack 默认** | 主基线 |
| T3 | ByteTrack + 参数调优 | `track_thresh / match_thresh / track_buffer` 网格（按 fps 换算） |
| T4 | ByteTrack + CMC | 车载自车运动补偿（BDD/KITTI 必要） |
| T5 | BoT-SORT（ReID 开启） | 外观关联上界（代价最高） |

对比维度：
- **ID 维持**：IDSW、IDF1、每 GT 轨迹平均 ID 数（理想=1）、Frag（轨迹中断次数）
- **速度**：端到端 FPS、每帧耗时 P50/P95、显存峰值
- **轨迹质量**：轨迹数、轨迹长度分布、≥30 帧轨迹占比

## 5. 轨迹输出格式

```json
{"frame": 42, "ts": 8.4, "id": 7, "cls": "car", "conf": 0.88,
 "bbox": [620, 310, 705, 372], "center": [662.5, 341.0],
 "vel_px": [4.2, -0.8], "speed_px_s": 21.4, "age": 18, "state": "tracked"}
```
逐轨迹汇总：`id, cls, start_frame, end_frame, 持续时长, 轨迹总长(px), 平均/最大速度, 是否间断`。

## 6. 执行步骤（约 2~3 小时）

| 步 | 内容 | 时间 |
|---|---|---|
| 1 | 装 torch cu124 + ultralytics，验证 GPU 可用 | 15 min |
| 2 | 拉 1 条 BDD val 序列（~40 MB，直取单帧） | 2 min |
| 3 | `track_video.py` 跑通 YOLO11+ByteTrack，出 mp4 + jsonl | 20 min |
| 4 | `eval_single_seq.py` 用 GT 算 IDSW/IDF1（单序列） | 20 min |
| 5 | `compare_trackers.py` 跑 T1~T5，出对比表 | 40 min |
| 6 | 轨迹图 + 报告 | 30 min |

## 7. 与之前大计划的取舍

- **保留**：ID 维持、ID 切换率、速度对比、轨迹生成、V2X 消息输出 —— 全部保留，只是从"多数据集"降到"单视频"。
- **放弃**：BDD 40 条序列统计、MOT17 压力测试、KITTI 米制轨迹、策略迁移矩阵 —— 这些是"论文级结论"，单视频演示不需要。
- **可选回补**：若之后要写论文/答辩加强，再按 `docs/PLAN.md` 扩大规模，代码结构完全复用。
