#!/usr/bin/env python3
"""检测质量评测（单帧级 precision/recall，IoU=0.5），用于"微调前 vs 微调后"对比。

- 支持 COCO 预训练模型（自动把 COCO 类别映射到 BDD 8 类）
- 支持微调后的 BDD 8 类模型（类别名直接对应）
- 只评测 GT 中属于 8 类的框；other person/other vehicle/trailer 忽略
- 按类别输出 TP/FP/FN/Precision/Recall，并给出不同 conf 阈值的扫描

用法:
  python eval_det.py --model yolo11s.pt --seqs-json data/meta/split.json --group eval --limit-seq 3
  python eval_det.py --model runs/detect/runs/bdd_ft/weights/best.pt --group eval --confs 0.05,0.15,0.25
"""
import argparse
import collections
import glob
import json
import os
import time
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

BDD8 = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
# COCO 名称 -> BDD 名称
COCO2BDD = {"person": "pedestrian", "bicycle": "bicycle", "car": "car",
            "motorcycle": "motorcycle", "bus": "bus", "truck": "truck",
            "train": "train", "rider": "rider"}


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A, B = a[:, None, :], b[None, :, :]
    xx1 = np.maximum(A[..., 0], B[..., 0]); yy1 = np.maximum(A[..., 1], B[..., 1])
    xx2 = np.minimum(A[..., 2], B[..., 2]); yy2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ua = ((A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
          + (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1]) - inter)
    return inter / np.maximum(ua, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqs-json", default="data/meta/split.json")
    ap.add_argument("--group", default="eval")
    ap.add_argument("--limit-seq", type=int, default=0, help="只跑前 N 条序列（0=全部）")
    ap.add_argument("--limit-frames", type=int, default=0, help="每条序列只用前 N 帧（0=全部）")
    ap.add_argument("--confs", default="0.05,0.15,0.25")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--labels", default="data/raw/bdd100k/box_track_labels_trainval.zip")
    ap.add_argument("--frames-root", default="data/raw/bdd100k/frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    is_bdd = "pedestrian" in set(names.values())
    # 把模型类别索引 -> BDD 类别名
    idx2bdd = {}
    for i, n in names.items():
        b = n if is_bdd else COCO2BDD.get(n)
        if b in BDD8:
            idx2bdd[i] = b
    print(f"模型: {args.model} | {'BDD8 微调' if is_bdd else 'COCO 预训练'} | 可评测类别: {sorted(set(idx2bdd.values()))}")

    seqs = json.load(open(args.seqs_json))[args.group]
    if args.limit_seq:
        seqs = seqs[:args.limit_seq]
    lab = zipfile.ZipFile(args.labels)

    confs = [float(c) for c in args.confs.split(",")]
    result = {c: collections.defaultdict(lambda: dict(tp=0, fp=0, fn=0, gt=0)) for c in confs}
    t0 = time.perf_counter()
    n_img = 0
    for seq in seqs:
        gt = None
        for sp in ("train", "val"):
            try:
                gt = json.loads(lab.read(f"bdd100k/labels/box_track_20/{sp}/{seq}.json"))
                break
            except KeyError:
                continue
        if gt is None:
            print(f"  [跳过] 无标签 {seq}")
            continue
        files = sorted(glob.glob(os.path.join(args.frames_root, seq, "*.jpg")))
        if args.limit_frames:
            files = files[:args.limit_frames]
        gmap = {fr["name"]: fr for fr in gt}
        for f in files:
            fr = gmap.get(os.path.basename(f))
            if fr is None:
                continue
            g = [lb for lb in (fr.get("labels") or []) if lb["category"] in BDD8]
            gb = np.array([[x["box2d"]["x1"], x["box2d"]["y1"], x["box2d"]["x2"], x["box2d"]["y2"]]
                           for x in g]).reshape(-1, 4)
            gc = [x["category"] for x in g]
            r = model.predict(f, imgsz=args.imgsz, conf=min(confs), classes=list(idx2bdd),
                              half=True, verbose=False)[0]
            b = r.boxes
            pb_all = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            pc_all = [idx2bdd[int(c)] for c in b.cls.cpu().numpy()] if len(b) else []
            ps_all = b.conf.cpu().numpy() if len(b) else np.zeros(0)
            n_img += 1
            for cf in confs:
                keep = ps_all >= cf
                pb, pc = pb_all[keep], [c for c, k in zip(pc_all, keep) if k]
                for name in set(gc):
                    gi = [i for i, c in enumerate(gc) if c == name]
                    pi = [i for i, c in enumerate(pc) if c == name]
                    result[cf][name]["gt"] += len(gi)
                    if not gi or not pi:
                        result[cf][name]["fn"] += len(gi)
                        result[cf][name]["fp"] += len(pi)
                        continue
                    M = iou_matrix(pb[pi], gb[gi])
                    cost = np.where(M >= 0.5, -M, 1e6)
                    ri, ci = linear_sum_assignment(cost)
                    m = int((cost[ri, ci] < 1e5).sum())
                    result[cf][name]["tp"] += m
                    result[cf][name]["fp"] += len(pi) - m
                    result[cf][name]["fn"] += len(gi) - m

    dt = time.perf_counter() - t0
    print(f"\n共 {n_img} 帧，耗时 {dt:.0f}s（{n_img/dt:.1f} FPS 含读图）\n")
    for cf in confs:
        print(f"--- conf={cf} ---")
        print(f"{'类别':<12}{'GT':>7}{'TP':>7}{'FP':>7}{'FN':>7}{'召回':>8}{'精确':>8}")
        T = dict(tp=0, fp=0, fn=0, gt=0)
        for name in BDD8:
            s = result[cf].get(name)
            if not s or s["gt"] == 0:
                continue
            rec = s["tp"] / max(s["gt"], 1)
            pre = s["tp"] / max(s["tp"] + s["fp"], 1)
            print(f"{name:<12}{s['gt']:>7}{s['tp']:>7}{s['fp']:>7}{s['fn']:>7}{rec:>8.0%}{pre:>8.0%}")
            for k in T:
                T[k] += s[k]
        if T["gt"]:
            print(f"{'合计':<12}{T['gt']:>7}{T['tp']:>7}{T['fp']:>7}{T['fn']:>7}"
                  f"{T['tp']/T['gt']:>8.0%}{T['tp']/max(T['tp']+T['fp'],1):>8.0%}")
        print()
    if args.out:
        json.dump({str(c): {k: dict(v) for k, v in result[c].items()} for c in confs},
                  open(args.out, "w"), indent=1)
        print(f"结果已存 {args.out}")


if __name__ == "__main__":
    main()
