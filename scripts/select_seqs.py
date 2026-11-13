#!/usr/bin/env python3
"""BDD100K 序列筛选：只读标签（无需图片），给 1600 条跟踪序列打分并分层抽样。

打分维度（全部来自 box_track_20 标签）：
  车辆密度 / 行人密度 / 遮挡比例 / 截断比例 / 轨迹长度分布 / 类别多样性 / 轨迹数
筛选约束：
  训练用序列只能来自 train 划分（1400 条），且必须排除评测用 val 序列（防泄漏）
输出：
  data/meta/bdd_seq_selection.csv
"""
import argparse
import collections
import json
import os
import zipfile

FOCUS = ["car", "truck", "bus", "pedestrian", "rider", "bicycle", "motorcycle"]


def stats_of(seq_labels):
    n_frames = len(seq_labels)
    c = collections.Counter()
    occ = trunc = 0
    tot = 0
    track_len = collections.defaultdict(int)
    for fr in seq_labels:
        for lb in (fr.get("labels") or []):
            c[lb["category"]] += 1
            tot += 1
            a = lb.get("attributes") or {}
            occ += 1 if a.get("occluded") else 0
            trunc += 1 if a.get("truncated") else 0
            track_len[lb["id"]] += 1
    lens = sorted(track_len.values())
    veh = sum(c[k] for k in ("car", "truck", "bus", "motorcycle", "bicycle"))
    per = sum(c[k] for k in ("pedestrian", "rider"))
    return {
        "frames": n_frames, "boxes": tot,
        "car": c["car"], "truck": c["truck"], "bus": c["bus"],
        "pedestrian": c["pedestrian"], "rider": c["rider"],
        "bicycle": c["bicycle"], "motorcycle": c["motorcycle"],
        "veh_boxes": veh, "per_boxes": per,
        "veh_per_frame": round(veh / max(n_frames, 1), 2),
        "per_per_frame": round(per / max(n_frames, 1), 2),
        "occ_ratio": round(occ / max(tot, 1), 3),
        "trunc_ratio": round(trunc / max(tot, 1), 3),
        "n_tracks": len(track_len),
        "track_len_median": lens[len(lens) // 2] if lens else 0,
        "long_track_ratio": round(sum(1 for x in lens if x >= 100) / max(len(lens), 1), 3),
        "n_classes": sum(1 for k in FOCUS if c[k] > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/bdd100k/box_track_labels_trainval.zip")
    ap.add_argument("--out", default="data/meta/bdd_seq_selection.csv")
    ap.add_argument("--n-train", type=int, default=30, help="要选的训练序列数")
    ap.add_argument("--exclude", default="b1e9ee0e-67e26f2e",
                    help="要排除的序列（评测用，逗号分隔）")
    args = ap.parse_args()
    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    z = zipfile.ZipFile(args.zip)
    rows = []
    for split in ("train", "val"):
        for name in [n for n in z.namelist()
                     if n.startswith(f"bdd100k/labels/box_track_20/{split}/") and n.endswith(".json")]:
            seq = os.path.basename(name)[:-5]
            d = json.loads(z.read(name))
            row = {"seq": seq, "split": split}
            row.update(stats_of(d))
            rows.append(row)
    print(f"总序列 {len(rows)}（train {sum(r['split']=='train' for r in rows)} / "
          f"val {sum(r['split']=='val' for r in rows)}）")

    cand = [r for r in rows if r["split"] == "train" and r["seq"] not in exclude]
    print(f"候选训练序列 {len(cand)}（已排除评测序列 {exclude}）")

    # 分层抽样：4 个桶，按桶内综合分排序取前若干个
    def score_car(r):      # 车辆密集
        return r["veh_per_frame"] * (1 + r["occ_ratio"]) * (0.5 + min(r["track_len_median"], 200) / 200)

    def score_ped(r):      # 行人密集
        return r["per_per_frame"] * (1 + r["occ_ratio"]) * (0.5 + min(r["track_len_median"], 200) / 200)

    def score_long(r):     # 长轨迹（ID 维持可测性）
        return r["long_track_ratio"] * 10 + r["track_len_median"] / 100

    def score_diverse(r):  # 类别多样性
        return r["n_classes"] * 2 + r["n_tracks"] / 50

    buckets = [("car_dense", score_car), ("ped_dense", score_ped),
               ("long_track", score_long), ("diverse", score_diverse)]
    picked, seen = [], set()
    per_bucket = max(1, args.n_train // len(buckets))
    for bname, fn in buckets:
        ranked = sorted([r for r in cand if r["seq"] not in seen], key=fn, reverse=True)
        n = 0
        for r in ranked:
            if n >= per_bucket:
                break
            r["bucket"] = bname
            r["score"] = round(fn(r), 3)
            picked.append(r)
            seen.add(r["seq"])
            n += 1
    # 不足则按车辆密度补齐
    for r in sorted(cand, key=score_car, reverse=True):
        if len(picked) >= args.n_train:
            break
        if r["seq"] in seen:
            continue
        r["bucket"] = "car_dense_fill"
        r["score"] = round(score_car(r), 3)
        picked.append(r)
        seen.add(r["seq"])

    cols = ["seq", "split", "bucket", "score", "frames", "veh_per_frame", "per_per_frame",
            "occ_ratio", "trunc_ratio", "n_tracks", "track_len_median", "long_track_ratio",
            "n_classes", "car", "truck", "bus", "pedestrian", "rider", "bicycle", "motorcycle"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in picked:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
        f.write("# 评测序列(排除) " + ",".join(sorted(exclude)) + "\n")

    print(f"\n选出 {len(picked)} 条训练序列 -> {args.out}")
    print(f"{'seq':<22}{'bucket':<16}{'车/帧':>7}{'人/帧':>7}{'遮挡':>7}{'轨迹中位':>9}{'长轨迹':>8}{'类别':>5}")
    for r in picked:
        print(f"{r['seq']:<22}{r['bucket']:<16}{r['veh_per_frame']:>7}{r['per_per_frame']:>7}"
              f"{r['occ_ratio']:>7}{r['track_len_median']:>9}{r['long_track_ratio']:>8}{r['n_classes']:>5}")


if __name__ == "__main__":
    main()
