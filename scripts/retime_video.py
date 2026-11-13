#!/usr/bin/env python3
"""把低帧率 mp4 变成"平滑播放"版：每帧重复 N 次，输出 fps = 源 fps × N。

只改播放节奏，不改画面内容，也不影响任何跟踪/指标结果。

用法:
  python retime_video.py --src out/a_5fps.mp4 --dup 3 --out out/a_15fps.mp4
"""
import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dup", type=int, default=3)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = args.out or os.path.splitext(args.src)[0] + f"_{int(fps*args.dup)}fps.mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps * args.dup, (w, h))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        for _ in range(args.dup):
            vw.write(fr)
        n += 1
    cap.release(); vw.release()
    print(f"{out}  {n} 帧 ×{args.dup} → 播放 {fps*args.dup:.0f} fps")


if __name__ == "__main__":
    main()
