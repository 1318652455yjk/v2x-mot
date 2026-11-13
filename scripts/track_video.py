#!/usr/bin/env python3
"""YOLO11 + ByteTrack 单视频目标跟踪（逐帧可控循环，便于分段计时与轨迹导出）

用法示例
  # 图片序列目录（BDD100K 跟踪序列）
  python track_video.py --source data/raw/bdd100k/frames/b1e9ee0e-67e26f2e \
      --model yolo11s.pt --fps 5 --imgsz 960

  # 换跟踪策略
  python track_video.py --source ... --tracker botsort.yaml --tag botsort

输出（默认 out/ 下）
  <name>_<tag>.mp4          带 ID 与轨迹尾迹的视频
  <name>_<tag>.jsonl        逐帧逐目标轨迹记录
  <name>_<tag>_tracks.csv   逐轨迹汇总
  <name>_<tag>_stats.json   速度/统计
"""
import argparse
import collections
import glob
import json
import os
import statistics
import time

import cv2
import numpy as np

# COCO 类别索引 -> 名称（只保留交通相关）
TRAFFIC_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                   5: "bus", 7: "truck"}
COLORS = {
    "person": (0, 0, 255), "pedestrian": (0, 0, 255), "bicycle": (255, 255, 0),
    "car": (0, 200, 0), "motorcycle": (128, 0, 255), "bus": (255, 128, 0),
    "truck": (0, 128, 255), "rider": (255, 0, 255), "train": (128, 128, 0),
}


def collect_frames(source):
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                       glob.glob(os.path.join(source, "*.png")))
        return files, None
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return [source], fps, n


def read_source(source):
    """返回 (generator of (idx, frame), fps, total)"""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                       glob.glob(os.path.join(source, "*.png")))
        if not files:
            raise SystemExit(f"目录里没有图片: {source}")
        return files, None, len(files)

    def gen():
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield i, fr
            i += 1
        cap.release()

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return gen(), fps, n


def frames_iter(source):
    """统一迭代器: yield (frame_idx, bgr_image)"""
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, "*.jpg")) +
                       glob.glob(os.path.join(source, "*.png")))
        if not files:
            raise SystemExit(f"目录里没有图片: {source}")
        for i, p in enumerate(files):
            img = cv2.imread(p)
            if img is None:
                raise SystemExit(f"读不了图片: {p}")
            yield i, img
        return
    cap = cv2.VideoCapture(source)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield i, fr
        i += 1
    cap.release()


def pct(v, p):
    return float(np.percentile(v, p)) if v else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="图片序列目录 或 视频文件")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml / botsort.yaml / 自定义")
    ap.add_argument("--tag", default=None, help="输出文件名后缀，默认取 tracker 名")
    ap.add_argument("--fps", type=float, default=None, help="源帧率（图片序列必须给，用于速度换算）")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classes", default=None,
                    help="类别索引，逗号分隔；默认：COCO 预训练用 0,1,2,3,5,7，微调模型用全部")
    ap.add_argument("--half", action="store_true", help="FP16 推理（GPU 上更快）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--trail", type=int, default=30, help="轨迹尾迹长度（帧）")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    src_name = os.path.basename(args.source.rstrip("/")) or "video"
    tag = args.tag or os.path.splitext(os.path.basename(args.tracker))[0]
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"{src_name}_{tag}")

    fps = args.fps
    if fps is None and not os.path.isdir(args.source):
        cap = cv2.VideoCapture(args.source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    fps = fps or 30.0

    model = YOLO(args.model)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    is_bdd = "pedestrian" in set(names.values())          # 微调后的 BDD 8 类模型
    if args.classes is None:
        classes = list(names.keys()) if is_bdd else [0, 1, 2, 3, 5, 7]
    else:
        classes = [int(c) for c in args.classes.split(",") if c.strip() != ""]
    print(f"模型类别({len(names)}): {list(names.values())[:10]} | 跟踪类别: {classes}")

    tracks = collections.defaultdict(list)   # id -> [(frame, cls, conf, box, center)]
    hist = collections.defaultdict(collections.deque)
    lat, det_n, t_start = [], [], time.perf_counter()

    jl = open(base + ".jsonl", "w", encoding="utf-8")
    writer = None
    n_frames = 0
    for idx, img in frames_iter(args.source):
        t0 = time.perf_counter()
        res = model.track(img, persist=True, tracker=args.tracker,
                          conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                          classes=classes, half=args.half, device=args.device,
                          verbose=False)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)

        recs = []
        boxes = res[0].boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(confs)
            det_n.append(len(confs))
            for (x1, y1, x2, y2), cf, cl, tid in zip(xyxy, confs, clss, ids):
                cname = names.get(int(cl), TRAFFIC_CLASSES.get(int(cl), str(cl)))
                cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
                prev = tracks[tid][-1][4] if tracks.get(tid) else None
                vx = vy = 0.0
                if prev is not None:
                    vx, vy = cx - prev[0], cy - prev[1]
                rec = {"frame": idx, "ts": round(idx / fps, 3), "id": int(tid),
                       "cls": cname, "conf": round(float(cf), 4),
                       "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
                       "center": [round(cx, 1), round(cy, 1)],
                       "vel_px": [round(vx, 2), round(vy, 2)],
                       "speed_px_s": round((vx ** 2 + vy ** 2) ** 0.5 * fps, 2),
                       "state": "tracked"}
                recs.append(rec)
                if tid >= 0:
                    tracks[tid].append((idx, cname, float(cf), (x1, y1, x2, y2), (cx, cy)))
                    hist[tid].append((int(cx), int(cy)))
                    if len(hist[tid]) > args.trail:
                        hist[tid].popleft()
                jl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not args.no_video:
            vis = img.copy()
            if args.trail > 0:
                for tid, pts in hist.items():
                    if len(pts) > 1:
                        col = COLORS.get(tracks[tid][0][1], (200, 200, 200))
                        cv2.polylines(vis, [np.array(pts, np.int32)], False, col, 2)
            for r in recs:
                x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
                col = COLORS.get(r["cls"], (200, 200, 200))
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                label = f"{r['cls']}#{r['id']} {r['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, max(y1 - th - 4, 0)), (x1 + tw + 2, y1), col, -1)
                cv2.putText(vis, label, (x1 + 1, max(y1 - 4, th)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, f"frame {idx} | ids {len(recs)} | {lat[-1]:.1f} ms",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            if writer is None:
                h, w = vis.shape[:2]
                out_video = base + ".mp4"
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(vis)

        n_frames += 1
        if n_frames % 50 == 0:
            print(f"  {n_frames} 帧 … 已建 ID {len(tracks)}")

    jl.close()
    if writer is not None:
        writer.release()
    wall = time.perf_counter() - t_start

    # 逐轨迹汇总
    rows = []
    for tid, seq in tracks.items():
        fr = [s[0] for s in seq]
        pts = [s[4] for s in seq]
        dist = sum(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
                   for i in range(1, len(pts)))
        gaps = [b - a for a, b in zip(fr, fr[1:]) if b - a > 1]
        rows.append({"id": tid, "cls": seq[0][1], "start": min(fr), "end": max(fr),
                     "frames": len(seq), "span": max(fr) - min(fr) + 1,
                     "path_px": round(dist, 1), "avg_speed_px_s": round(dist / max(len(seq) - 1, 1) * fps, 1),
                     "gaps": len(gaps), "mean_conf": round(statistics.mean(s[2] for s in seq), 3)})
    rows.sort(key=lambda r: -r["frames"])
    csv_path = base + "_tracks.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,cls,start,end,frames,span,path_px,avg_speed_px_s,gaps,mean_conf\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in
                             ("id", "cls", "start", "end", "frames", "span",
                              "path_px", "avg_speed_px_s", "gaps", "mean_conf")) + "\n")

    lens = [r["frames"] for r in rows]
    stats = {
        "source": args.source, "tracker": args.tracker, "model": args.model,
        "frames": n_frames, "fps_src": fps,
        "wall_time_s": round(wall, 2),
        "fps_e2e": round(n_frames / wall, 2),
        "latency_ms_p50": round(pct(lat, 50), 2),
        "latency_ms_p95": round(pct(lat, 95), 2),
        "latency_ms_mean": round(statistics.mean(lat), 2) if lat else 0,
        "num_track_ids": len(tracks),
        "det_per_frame_mean": round(statistics.mean(det_n), 2) if det_n else 0,
        "det_per_frame_max": max(det_n) if det_n else 0,
        "track_len_median": statistics.median(lens) if lens else 0,
        "tracks_ge_30_frames": sum(1 for x in lens if x >= 30),
        "tracks_ge_60_frames": sum(1 for x in lens if x >= 60),
        "tracks_ge_120_frames": sum(1 for x in lens if x >= 120),
        "ids_per_class": dict(sorted(collections.Counter(r["cls"] for r in rows).items())),
        "model_classes": list(names.values()),
    }
    with open(base + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 结果 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n输出: {base}.mp4 / .jsonl / _tracks.csv / _stats.json")


if __name__ == "__main__":
    main()
