#!/usr/bin/env python3
"""跟踪速度分段基准：preprocess / inference / 关联 / 后处理 各段耗时 + 端到端 FPS。

用法:
  python bench_speed.py --source data/raw/bdd100k/frames/b1e9ee0e-67e26f2e \
      --model yolo11s.pt --imgsz 1280 --warmup 10 --repeat 203
输出:
  out/fig/speed_<tag>.png   分段耗时条形图 + 帧耗时分布
  out/speed_<tag>.json      原始统计
"""
import argparse
import glob
import json
import os
import statistics
import time

import cv2
import numpy as np

matplotlib_ok = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib_ok = False


def pct(v, p):
    return float(np.percentile(v, p)) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--tracker", default="configs/track/bt_5fps.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=0, help="0=跑完整个序列")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    files = sorted(glob.glob(os.path.join(args.source, "*.jpg")))
    if not files:
        raise SystemExit("没有图片")
    if args.repeat:
        files = files[:args.repeat]
    model = YOLO(args.model)
    names = model.names
    classes = list(names.keys()) if "pedestrian" in set(names.values()) else [0, 1, 2, 3, 5, 7]

    # 预热
    for f in files[:args.warmup]:
        model.track(cv2.imread(f), persist=True, tracker=args.tracker, conf=args.conf,
                    imgsz=args.imgsz, classes=classes, half=args.half, verbose=False)
    model.predictor.trackers[0].reset()

    t_read, t_pre, t_inf, t_post, t_total = [], [], [], [], []
    n_obj = []
    for f in files:
        t0 = time.perf_counter()
        img = cv2.imread(f)
        t1 = time.perf_counter()
        r = model.track(img, persist=True, tracker=args.tracker, conf=args.conf,
                        imgsz=args.imgsz, classes=classes, half=args.half, verbose=False)
        t2 = time.perf_counter()
        speed = r[0].speed if hasattr(r[0], "speed") else {}
        # ultralytics 给出的分段（ms）
        t_pre.append(speed.get("preprocess", 0.0))
        t_inf.append(speed.get("inference", 0.0))
        t_post.append(speed.get("postprocess", 0.0))
        n_obj.append(len(r[0].boxes) if r[0].boxes is not None else 0)
        t_read.append((t1 - t0) * 1000)
        t_total.append((t2 - t0) * 1000)

    assoc = [max(t_post[i] + (t_total[i] - t_read[i] - t_pre[i] - t_inf[i] - t_post[i]), 0.001)
             for i in range(len(t_total))] if t_total else []
    res = {
        "source": args.source, "model": args.model, "imgsz": args.imgsz,
        "frames": len(files),
        "fps_e2e": round(1000 / statistics.mean(t_total), 2),
        "ms_read_p50": round(pct(t_read, 50), 2),
        "ms_preprocess_p50": round(pct(t_pre, 50), 2),
        "ms_inference_p50": round(pct(t_inf, 50), 2),
        "ms_postprocess_p50": round(pct(t_post, 50), 2),
        "ms_total_p50": round(pct(t_total, 50), 2),
        "ms_total_p95": round(pct(t_total, 95), 2),
        "objs_mean": round(statistics.mean(n_obj), 1) if n_obj else 0,
        "objs_max": max(n_obj) if n_obj else 0,
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }
    tag = args.tag or f"{os.path.basename(args.source.rstrip('/'))}_{os.path.splitext(args.model)[0]}"
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.out_dir, f"speed_{tag}.json"), "w"), indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))

    if matplotlib_ok:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=110)
        segs = ["读图", "预处理", "推理", "后处理/关联"]
        vals = [res["ms_read_p50"], res["ms_preprocess_p50"],
                res["ms_inference_p50"], res["ms_postprocess_p50"]]
        axes[0].bar(segs, vals, color=["#8c8c8c", "#1f77b4", "#d62728", "#2ca02c"])
        for i, v in enumerate(vals):
            axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        axes[0].set_ylabel("ms / frame (P50)")
        axes[0].set_title(f"Stage latency @ imgsz={args.imgsz} ({res['fps_e2e']} FPS e2e)")
        axes[0].grid(alpha=0.3, axis="y")
        axes[1].hist(t_total, bins=25, color="#9467bd", alpha=0.85)
        axes[1].axvline(res["ms_total_p50"], color="k", ls="--", label=f"P50 {res['ms_total_p50']} ms")
        axes[1].axvline(res["ms_total_p95"], color="r", ls=":", label=f"P95 {res['ms_total_p95']} ms")
        axes[1].set_xlabel("frame latency (ms)"); axes[1].set_ylabel("#frames")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        p = os.path.join(args.out_dir, "fig", f"speed_{tag}.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        print("图:", p)


if __name__ == "__main__":
    main()
