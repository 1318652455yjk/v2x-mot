#!/usr/bin/env python3
"""单序列跟踪评测（BDD100K GT，MOTChallenge 风格指标）

指标：IDSW / IDF1 / IDP / IDR / MOTA / MOTP / Frag / MT / ML / 每 GT 轨迹平均 ID 数
用法:
  python eval_tracks.py --jsonl out/<name>_bt.jsonl --seq b1e9ee0e-67e26f2e
"""
import argparse
import collections
import json
import zipfile

import numpy as np
from scipy.optimize import linear_sum_assignment

# BDD 类别 -> 检测类别（只评测这几类；other person/other vehicle/trailer 忽略）
# GT 类别 -> 检测类别；同一 GT 类别允许多种检测名（COCO 的 person / 微调模型的 pedestrian）
GT2DET = {"pedestrian": ("person", "pedestrian"), "car": ("car",), "truck": ("truck",),
          "bus": ("bus",), "motorcycle": ("motorcycle",), "bicycle": ("bicycle",)}


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A, B = a[:, None, :], b[None, :, :]
    xx1 = np.maximum(A[..., 0], B[..., 0])
    yy1 = np.maximum(A[..., 1], B[..., 1])
    xx2 = np.minimum(A[..., 2], B[..., 2])
    yy2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ua = ((A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
          + (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1]) - inter)
    return inter / np.maximum(ua, 1e-9)


def load_gt(zip_path, seq):
    with zipfile.ZipFile(zip_path) as z:
        for split in ("train", "val"):
            try:
                data = json.loads(z.read(f"bdd100k/labels/box_track_20/{split}/{seq}.json"))
            except KeyError:
                continue
            out = {}
            for i, fr in enumerate(data):
                items = []
                for lb in (fr.get("labels") or []):
                    if lb["category"] not in GT2DET:
                        continue
                    b = lb["box2d"]
                    items.append((int(lb["id"]), lb["category"],
                                  (b["x1"], b["y1"], b["x2"], b["y2"]),
                                  lb["attributes"]))
                out[i] = items
            return out
    raise FileNotFoundError(seq)


def load_pred(jsonl):
    out = collections.defaultdict(list)
    for line in open(jsonl, encoding="utf-8"):
        r = json.loads(line)
        out[r["frame"]].append((r["id"], r["cls"], r["bbox"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--gt-zip", default="data/raw/bdd100k/box_track_labels_trainval.zip")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    gt_all = load_gt(args.gt_zip, args.seq)
    pred = load_pred(args.jsonl)
    frames = sorted(gt_all)

    gt_tid2cls = {}
    for f in frames:
        for tid, cls, _, _ in gt_all[f]:
            gt_tid2cls.setdefault(tid, cls)

    n_gt = n_pred = tp = fp = fn = idsw = 0
    iou_sum = 0.0
    frag = collections.Counter()
    gt_hit_frames = collections.defaultdict(set)
    gt_tracks = collections.defaultdict(set)       # gt id -> set(frame)
    pair = collections.Counter()                  # (gt id, pred id) -> 匹配帧数
    last_match = {}                               # gt id -> pred id
    gt_seen = collections.defaultdict(set)        # gt id -> 已出现帧

    for f in frames:
        g = gt_all[f]
        p = pred.get(f, [])
        n_gt += len(g)
        n_pred += len(p)
        gb = np.array([x[2] for x in g]).reshape(-1, 4)
        pb = np.array([x[2] for x in p]).reshape(-1, 4)
        M = iou_matrix(pb, gb)
        ok = M >= args.iou
        # 类别同组 + Hungarian
        def same(a, b):
            return b in GT2DET.get(a, (a,))
        same_cls = np.array([[same(g[j][1], p[i][1]) for j in range(len(g))]
                             for i in range(len(p))]).reshape(len(p), len(g)) \
            if (p and g) else np.zeros((len(p), len(g)), bool)
        cost = np.where(ok & same_cls, -M, 1e6)
        matched_p = set()
        if cost.size:
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] > 1e5:
                    continue
                matched_p.add(i)
                pid, pcls, _ = p[i]
                gid, gcls, _, _ = g[j]
                tp += 1
                iou_sum += M[i, j]
                pair[(gid, pid)] += 1
                gt_hit_frames[gid].add(f)
                gt_tracks[gid].add(f)
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                    frag[gid] += 1
                last_match[gid] = pid
        fp += len(p) - len(matched_p)
        fn += len(g) - len(matched_p)

    # 碎片化：GT 轨迹被覆盖的帧出现断档
    for gid, fs in gt_tracks.items():
        fs = sorted(fs)
        gaps = sum(1 for a, b in zip(fs, fs[1:]) if b - a > 1)
        frag[gid] += gaps

    # IDF1（全局 ID 级匹配）
    gtids = sorted({k[0] for k in pair})
    pids = sorted({k[1] for k in pair})
    if gtids and pids:
        C = np.zeros((len(gtids), len(pids)))
        gi = {g: i for i, g in enumerate(gtids)}
        pi = {p: i for i, p in enumerate(pids)}
        for (g, p), c in pair.items():
            C[gi[g], pi[p]] = c
        r, c = linear_sum_assignment(-C)
        idtp = C[r, c].sum()
        idfp = n_pred - idtp
        idfn = n_gt - idtp
        idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1e-9)
        idp = idtp / max(idtp + idfp, 1e-9)
        idr = idtp / max(idtp + idfn, 1e-9)
    else:
        idtp = idf1 = idp = idr = 0.0

    # MT/ML：GT 轨迹被覆盖比例
    coverage = []
    for gid, cls in gt_tid2cls.items():
        tot = sum(1 for f in frames if any(x[0] == gid for x in gt_all[f]))
        cov = len(gt_hit_frames[gid]) / max(tot, 1)
        coverage.append(cov)
    mt = sum(1 for c in coverage if c >= 0.8)
    ml = sum(1 for c in coverage if c <= 0.2)

    ids_per_gt = collections.defaultdict(set)
    for (g, p) in pair:
        ids_per_gt[g].add(p)

    res = {
        "frames": len(frames),
        "gt_boxes": n_gt, "pred_boxes": n_pred,
        "tp": tp, "fp": fp, "fn": fn,
        "recall": round(tp / max(n_gt, 1), 4),
        "precision": round(tp / max(n_pred, 1), 4),
        "MOTA": round(1 - (fn + fp + idsw) / max(n_gt, 1), 4),
        "MOTP_iou": round(iou_sum / max(tp, 1), 4),
        "IDSW": idsw,
        "id_switch_rate": round(idsw / max(tp, 1), 4),
        "ID_keep_rate": round(1 - idsw / max(tp, 1), 4),
        "IDF1": round(idf1, 4), "IDP": round(idp, 4), "IDR": round(idr, 4),
        "Frag": int(sum(frag.values())),
        "MT": mt, "ML": ml, "n_gt_tracks": len(gt_tid2cls),
        "IDS_per_gt_track": round(np.mean([len(v) for v in ids_per_gt.values()]) if ids_per_gt else 0, 3),
        "n_pred_ids": len(pids),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
