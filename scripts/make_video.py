#!/usr/bin/env python3
"""把图片序列目录合成 mp4；可选：叠加 GT 框（检查数据正确性）。

用法:
  python make_video.py --frames data/raw/bdd100k/frames/b1e9ee0e-67e26f2e --fps 5
  python make_video.py --frames ... --fps 5 --dup 3 --out out/seq_15fps.mp4   # 播放更顺（只改播放速度）
  python make_video.py --frames ... --gt-seq b1e9ee0e-67e26f2e --gt-zip data/raw/bdd100k/box_track_labels_trainval.zip
"""
import argparse
import glob
import json
import os
import zipfile

import cv2

CLS_COLOR = {
    "car": (0, 200, 0), "truck": (0, 128, 255), "bus": (255, 128, 0),
    "pedestrian": (0, 0, 255), "rider": (255, 0, 255), "bicycle": (255, 255, 0),
    "motorcycle": (128, 0, 255), "train": (128, 128, 0),
}


def load_gt(zip_path, seq):
    with zipfile.ZipFile(zip_path) as z:
        for split in ("train", "val"):
            p = f"bdd100k/labels/box_track_20/{split}/{seq}.json"
            try:
                data = json.loads(z.read(p))
            except KeyError:
                continue
            return {fr["name"]: (fr.get("labels") or []) for fr in data}
    raise FileNotFoundError(f"{seq} 不在 {zip_path} 中")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--dup", type=int, default=1, help="每帧重复次数（只为播放更顺，不改跟踪结果）")
    ap.add_argument("--gt-seq", default=None)
    ap.add_argument("--gt-zip", default="data/raw/bdd100k/box_track_labels_trainval.zip")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    if not files:
        raise SystemExit(f"没有找到图片: {args.frames}")
    gt = load_gt(args.gt_zip, args.gt_seq) if args.gt_seq else None

    h, w = cv2.imread(files[0]).shape[:2]
    out = args.out or os.path.join(
        "out", os.path.basename(args.frames.rstrip("/")) + f"_{int(args.fps)}fps.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps * args.dup, (w, h))
    n_box = 0
    for f in files:
        img = cv2.imread(f)
        if gt is not None:
            for lb in gt.get(os.path.basename(f), []):
                b = lb["box2d"]
                p1 = (int(b["x1"]), int(b["y1"]))
                p2 = (int(b["x2"]), int(b["y2"]))
                color = CLS_COLOR.get(lb["category"], (200, 200, 200))
                cv2.rectangle(img, p1, p2, color, 2)
                cv2.putText(img, f"{lb['category']}#{lb['id']}", (p1[0], max(p1[1] - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                n_box += 1
        for _ in range(args.dup):
            vw.write(img)
    vw.release()
    print(f"输出: {out}  帧 {len(files)} × dup{args.dup} @ {args.fps}fps  "
          f"({w}x{h}) dup={args.dup} 实际播放 {args.fps*args.dup:.0f}fps  GT框 {n_box}")


if __name__ == "__main__":
    main()
