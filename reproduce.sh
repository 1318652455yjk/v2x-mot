#!/usr/bin/env bash
# 一键复现脚本（v1.0-yolo11）
#
#   bash reproduce.sh 0     # 0. 环境自检（不动数据）
#   bash reproduce.sh 1     # 1. 下载数据（约 10 GB，需能访问 hf-mirror）
#   bash reproduce.sh 2     # 2. 抽帧 + 划分 + 转 YOLO 格式
#   bash reproduce.sh 3     # 3. 训练微调（可选，已有 weights/ 可跳过）
#   bash reproduce.sh 4     # 4. 检测评测 + 跟踪 + 指标 + V2X + 图 + 测速
#   bash reproduce.sh 5     # 5. 复现性校验（与 repro/golden.json 对比）
#   bash reproduce.sh all   # 1→2→4→5（跳过训练，用已提供权重）
#
# 约定：所有路径相对项目根目录；PY 可用环境变量覆盖。
set -euo pipefail
PY="${PY:-$HOME/miniconda3/envs/py310/bin/python}"
STAGE="${1:-0}"
cd "$(dirname "$0")"

S_VEH=b251064f-e7a165fd      # 车辆密集（评测用固定序列）
S_PED=b1d22ed6-f1cac061      # 行人密集

sha_check() {
  echo "== 校验数据 zip 的 sha256 =="
  if ! sha256sum -c repro/env/sha256.txt 2>/dev/null | grep -v '^data\|^runs' ; then :; fi
  grep zip repro/env/sha256.txt | while read -r sum path; do
    [ -f "$path" ] || { echo "  缺失: $path"; continue; }
    cur=$(sha256sum "$path" | cut -d' ' -f1)
    [ "$cur" = "$sum" ] && echo "  OK   $path" || echo "  不一致 $path"
  done
}

stage_env() {
  echo "== 0. 环境自检 =="
  "$PY" - <<'EOF'
import torch, ultralytics, cv2
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} {torch.version.cuda}")
print(f"ultralytics {ultralytics.__version__} opencv {cv2.__version__}")
assert torch.cuda.is_available(), "CUDA 不可用：请安装 cu124 版 torch"
assert ultralytics.__version__ >= "8.3"
EOF
  echo "环境 OK（详见 repro/env/system.txt）"
}

stage_data() {
  echo "== 1. 下载数据 =="
  "$PY" scripts/hf_download.py images20-track-train-1.zip images20-track-val-1.zip \
        --out data/raw/bdd100k/zips --workers 8
  sha_check
}

stage_prep() {
  echo "== 2. 抽帧 + 划分 + 转 YOLO =="
  "$PY" scripts/prepare_local.py --n-train 40 --n-intval 10 --n-eval 10
  "$PY" scripts/bdd_to_yolo.py --seqs-file data/meta/yolo_train.csv --split train
  "$PY" scripts/bdd_to_yolo.py --seqs-file data/meta/yolo_val.csv   --split val
  echo "抽取完成：train 40 / intval 10 / eval 10"
}

stage_train() {
  echo "== 3. 微调 YOLO11（30 epoch，约 100 min @ RTX3050）=="
  "$PY" scripts/train_yolo.py --name bdd_ft --epochs 30
  echo "权重：runs/detect/runs/bdd_ft/weights/best.pt"
}

stage_run() {
  echo "== 4. 评测与产物 =="
  mkdir -p out out/multi out/fig

  echo "-- 4.1 检测评测（COCO vs 微调）--"
  "$PY" scripts/eval_det.py --model yolo11s.pt --group eval --limit-seq 3 \
        --confs 0.05,0.10 --imgsz 1280 | tee out/det_coco.txt
  [ -f runs/detect/runs/bdd_ft/weights/best.pt ] && \
    "$PY" scripts/eval_det.py --model runs/detect/runs/bdd_ft/weights/best.pt \
          --group eval --limit-seq 3 --confs 0.05,0.10 --imgsz 1280 | tee out/det_ft.txt

  echo "-- 4.2 策略对比（5 策略 × 3 序列）--"
  for t in bt_default bt_5fps bt_5fps_loose sort_like botsort_cmc; do
    for S in b1e9ee0e-67e26f2e b1d22ed6-f1cac061 b23c9e00-b425de1b; do
      "$PY" scripts/track_video.py --source "data/raw/bdd100k/frames/$S" \
            --model yolo11s.pt --fps 5 --imgsz 1280 --conf 0.05 --half \
            --tracker "configs/track/$t.yaml" --tag "$t" --no-video --out out/multi >/dev/null
      "$PY" scripts/eval_tracks.py --jsonl "out/multi/${S}_${t}.jsonl" --seq "$S" \
            --json-out "out/multi/${S}_${t}_eval.json" >/dev/null
    done
  done
  echo "  -> out/multi/strategy_summary.csv"

  echo "-- 4.3 最优配置产物（视频 / 轨迹 / V2X / 图）--"
  for S in "$S_VEH" "$S_PED" b1e9ee0e-67e26f2e; do
    "$PY" scripts/track_video.py --source "data/raw/bdd100k/frames/$S" \
          --model yolo11s.pt --fps 5 --imgsz 1280 --conf 0.05 --half \
          --tracker configs/track/botsort_cmc.yaml --tag best --out out
    "$PY" scripts/eval_tracks.py --jsonl "out/${S}_best.jsonl" --seq "$S" \
          --json-out "out/${S}_best_eval.json"
    "$PY" scripts/export_v2x.py --jsonl "out/${S}_best.jsonl" --fps 5 --assume-scale 1108
    "$PY" scripts/viz_traj.py --jsonl "out/${S}_best.jsonl" --seq "$S"
    "$PY" scripts/retime_video.py --src "out/${S}_best.mp4" --dup 3 >/dev/null
  done

  echo "-- 4.4 速度基准 --"
  "$PY" scripts/bench_speed.py --source "data/raw/bdd100k/frames/$S_VEH" \
        --model yolo11s.pt --imgsz 1280 --tracker configs/track/botsort_cmc.yaml --tag cmc
  "$PY" scripts/bench_speed.py --source "data/raw/bdd100k/frames/$S_VEH" \
        --model yolo11s.pt --imgsz 1280 --tracker configs/track/bt_5fps.yaml --tag bt_5fps
  echo "完成：见 out/ 与 out/fig/"
}

stage_verify() {
  echo "== 5. 复现性校验 =="
  "$PY" scripts/verify_repro.py
}

case "$STAGE" in
  0) stage_env ;;
  1) stage_data ;;
  2) stage_prep ;;
  3) stage_train ;;
  4) stage_run ;;
  5) stage_verify ;;
  all) stage_env; stage_data; stage_prep; stage_run; stage_verify ;;
  *) echo "用法: bash reproduce.sh {0|1|2|3|4|5|all}"; exit 1 ;;
esac
