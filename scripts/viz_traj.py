#!/usr/bin/env python3
"""轨迹可视化：轨迹平面图、目标数随时间、轨迹时长分布、ID 切换点标注。

用法:
  python viz_traj.py --jsonl out/b1e9ee0e-67e26f2e_final.jsonl --seq b1e9ee0e-67e26f2e
输出（out/fig/ 下）:
  traj_<tag>.png        图像平面轨迹（每条轨迹一色，含起点/终点）
  traj_stats_<tag>.png  目标数-时间曲线 + 轨迹时长直方图
  idsw_<tag>.png        与 GT 对比的 ID 切换点标注（需要 --seq）
"""
import argparse
import collections
import json
import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GT2DET = {"pedestrian": ("person", "pedestrian"), "car": ("car",), "truck": ("truck",),
          "bus": ("bus",), "motorcycle": ("motorcycle",), "bicycle": ("bicycle",)}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--seq", default=None, help="有 GT 时用于标注 ID 切换点")
    ap.add_argument("--labels", default="data/raw/bdd100k/box_track_labels_trainval.zip")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out-dir", default="out/fig")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or os.path.splitext(os.path.basename(args.jsonl))[0]
    recs = [json.loads(l) for l in open(args.jsonl, encoding="utf-8")]
    tracks = collections.defaultdict(list)
    per_frame = collections.Counter()
    for r in recs:
        tracks[r["id"]].append((r["frame"], r["center"], r["cls"]))
        per_frame[r["frame"]] += 1
    cls_of = {tid: v[0][2] for tid, v in tracks.items()}
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 轨迹平面图
    fig, ax = plt.subplots(figsize=(12, 7), dpi=110)
    cmap = plt.get_cmap("tab20")
    order = sorted(tracks, key=lambda t: -len(tracks[t]))
    for i, tid in enumerate(order):
        pts = np.array([p[1] for p in tracks[tid]])
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.2, color=cmap(i % 20), alpha=0.85)
        ax.plot(pts[0, 0], pts[0, 1], "o", ms=4, color=cmap(i % 20))
        ax.plot(pts[-1, 0], pts[-1, 1], "s", ms=4, color=cmap(i % 20))
        ax.annotate(f"{tid}", pts[len(pts) // 2], fontsize=6, color=cmap(i % 20))
    ax.set_xlim(0, 1280); ax.set_ylim(720, 0)      # y 轴向下（图像坐标）
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title(f"Trajectories ({len(tracks)} tracks, dot=start, square=end)")
    ax.grid(alpha=0.3)
    p1 = os.path.join(args.out_dir, f"traj_{tag}.png")
    fig.tight_layout(); fig.savefig(p1); plt.close(fig)

    # 2) 目标数-时间 + 轨迹时长直方图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=110)
    fs = sorted(per_frame)
    axes[0].plot(fs, [per_frame[f] for f in fs], lw=1.2, color="#1f77b4")
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("#objects")
    axes[0].set_title(f"Objects per frame (mean {np.mean([per_frame[f] for f in fs]):.1f})")
    axes[0].grid(alpha=0.3)
    lens = [len(v) for v in tracks.values()]
    axes[1].hist(lens, bins=30, color="#ff7f0e", alpha=0.85)
    axes[1].axvline(np.median(lens), color="k", ls="--", label=f"median {np.median(lens):.0f} frames")
    axes[1].set_xlabel("track length (frames)"); axes[1].set_ylabel("#tracks")
    axes[1].set_title(f"Track length distribution ({len(tracks)} tracks)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    p2 = os.path.join(args.out_dir, f"traj_stats_{tag}.png")
    fig.tight_layout(); fig.savefig(p2); plt.close(fig)

    print(f"输出: {p1}\n      {p2}")

    # 3) ID 切换点标注（有 GT 时）
    if args.seq:
        z = zipfile.ZipFile(args.labels)
        gt = None
        for sp in ("train", "val"):
            try:
                gt = json.loads(z.read(f"bdd100k/labels/box_track_20/{sp}/{args.seq}.json"))
                break
            except KeyError:
                continue
        if gt is None:
            print(f"[跳过] 找不到 {args.seq} 的 GT")
            return
        pred = collections.defaultdict(list)
        for r in recs:
            pred[r["frame"]].append(r)
        last = {}
        switches = []
        for fi, fr in enumerate(gt):
            g = [(int(l["id"]), l["category"],
                  (l["box2d"]["x1"], l["box2d"]["y1"], l["box2d"]["x2"], l["box2d"]["y2"]))
                 for l in (fr.get("labels") or []) if l["category"] in GT2DET]
            p = pred.get(fi, [])
            if not g or not p:
                continue
            best = {}
            for gid, gcls, gbox in g:
                cand = [(iou(gbox, pr["bbox"]), pr) for pr in p
                        if pr["cls"] in GT2DET.get(gcls, (gcls,))]
                cand = [c for c in cand if c[0] >= 0.5]
                if cand:
                    best[gid] = max(cand, key=lambda x: x[0])[1]
            for gid, pr in best.items():
                if gid in last and last[gid]["id"] != pr["id"]:
                    switches.append((fi, pr["center"], last[gid]["id"], pr["id"]))
                last[gid] = pr
        fig, ax = plt.subplots(figsize=(12, 7), dpi=110)
        for i, tid in enumerate(order):
            pts = np.array([p[1] for p in tracks[tid]])
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.0, color="0.75", alpha=0.8)
        if switches:
            sx = [s[1][0] for s in switches]; sy = [s[1][1] for s in switches]
            ax.scatter(sx, sy, marker="x", s=60, color="red", label=f"ID switch ×{len(switches)}")
            ax.legend()
        ax.set_xlim(0, 1280); ax.set_ylim(720, 0)
        ax.set_title(f"ID switch positions ({len(switches)} switches)")
        ax.grid(alpha=0.3)
        p3 = os.path.join(args.out_dir, f"idsw_{tag}.png")
        fig.tight_layout(); fig.savefig(p3); plt.close(fig)
        print(f"      {p3}  (ID 切换 {len(switches)} 次)")


if __name__ == "__main__":
    main()
