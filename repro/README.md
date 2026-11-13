# 复现指南（v1.0-yolo11）

## 1. 复现等级声明（重要）

| 环节 | 可复现性 | 说明 |
|---|---|---|
| 环境 | ✅ 完全 | `repro/env/{pip-freeze.txt, system.txt}` 固定版本 |
| 数据 | ✅ 完全 | 两个 zip 的 sha256 已固定（`repro/env/sha256.txt`），划分用固定脚本 |
| **推理 + 跟踪 + 指标** | ✅ **逐位一致** | `repro/golden.json` 为基准，容差内 PASS（实测完全相等） |
| 训练 | 🟡 近似 | 固定 `seed=0`，但 cuDNN/GPU 非确定性会让权重有微小差异；本仓库直接提供训练好的权重，**默认跳过训练** |
| 微调后指标 | 🟡 近似（±0.02） | 若自行重训，指标会落在 golden ±0.02 内 |

结论：**代码 + 数据 + 权重 + 基准四件套齐全，推理链路逐位可复现；训练链路近似可复现。**

## 2. 三步验证（已有数据时，约 3 分钟）

```bash
bash reproduce.sh 0     # 环境自检
bash reproduce.sh 5     # 复现性校验（重跑 2 条固定序列，与 golden 对比）
```

期望输出末尾：`总计 21/21 通过`

## 3. 从零完整复现（约 2.5 小时）

```bash
bash reproduce.sh all   # 环境 → 下载 → 抽帧/划分/转格式 → 评测+跟踪+产物 → 校验
```

分步：
| 命令 | 内容 | 数据量 / 耗时 |
|---|---|---|
| `bash reproduce.sh 1` | 从 hf-mirror 下 2 个 zip（8 线程分块） | 10.15 GB / ~20 min |
| `bash reproduce.sh 2` | 抽 60 条序列帧 + 三份划分 + 转 YOLO | 1.8 GB / ~5 min |
| `bash reproduce.sh 3` | 微调 YOLO11s（可选，权重已提供） | — / ~100 min |
| `bash reproduce.sh 4` | 检测评测 + 5 策略 × 3 序列 + V2X + 图 + 测速 | — / ~40 min |
| `bash reproduce.sh 5` | 校验 | — / ~3 min |

磁盘预算：原始 zip 10 GB + 帧 3.95 GB + YOLO 软链接（≈0）+ 产物 0.4 GB ≈ **15 GB**。

## 4. 黄金基准（`repro/golden.json`）

固定两条序列、固定配置（yolo11s COCO、imgsz 1280、conf 0.05、BoT-SORT+CMC）：

| 序列 | 场景 | IDF1 | MOTA | IDSW | 切换率 | ID 维持率 | 召回 |
|---|---|---|---|---|---|---|---|
| `b251064f-e7a165fd` | 车辆密集 | 0.5846 | 0.4818 | 96 | 3.07% | 96.93% | 0.5863 |
| `b1d22ed6-f1cac061` | 行人密集 | 0.4908 | 0.4762 | 272 | 7.60% | 92.40% | 0.5673 |

容差：IDF1/MOTA/recall ±0.02、切换率 ±0.005、IDSW ±10%。

## 5. 依赖（固定版本见 `repro/env/pip-freeze.txt`）

```bash
conda create -n v2xmot python=3.10 -y
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics==8.4.211 lap polars nvidia-ml-py \
    "opencv-python-headless==4.10.0.84" pandas scipy matplotlib tqdm remotezip
```

## 6. 常见坑（本项目实际踩过，按此排查）

| 症状 | 原因 | 处理 |
|---|---|---|
| 训练第 1 个 epoch 结束就崩 `No module named 'polars'` | ultralytics 8.4 读 results.csv 用 polars | `pip install polars` |
| `import cv2` 失败 / `libGL.so` 缺失 | 装了非 headless 版 opencv | 用 `opencv-python-headless` |
| 中文路径或 OpenCV 5.x 报错 | ultralytics 未适配 cv2 5.0 | 锁 4.10.0.84 |
| 换跟踪参数后 ID 数暴涨（45→319） | 把 `match_thresh` 当 IoU 门限（实际是 1−IoU 代价门限，越大越松） | 参考 `configs/track/bt_5fps.yaml` 注释 |
| 行人 ID 全丢 / 指标虚高 | `eval_tracks.py` 的类别别名不匹配（微调模型输出 `pedestrian`，COCO 输出 `person`） | 已用 `GT2DET` 别名组修复 |
| HF 下载 1.4 MB/s 太慢 | 单连接限速 | 用 `scripts/hf_download.py`（8 线程，16~68 MB/s）；或设置代理后重跑 |
| `pkill -f yolo11s` 把命令自己杀了 | 命令行里也含该字符串 | 用 `scripts/train_yolo.py` 启动，不要用 pkill 匹配模型名 |

## 7. 文件对应关系

```
reproduce.sh              一键复现入口（0~5 / all）
scripts/verify_repro.py   L1 环境 / L2 数据 / L3 结果 三层校验
repro/golden.json         黄金基准（9 项指标 × 2 序列）
repro/env/sha256.txt      数据 zip 与权重的 sha256
repro/env/pip-freeze.txt  233 个包的精确版本
repro/env/system.txt      OS / Python / GPU / CUDA / torch 版本
repro/out/                校验时产生的临时轨迹与指标（可删）
```
