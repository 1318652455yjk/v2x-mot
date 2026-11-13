#!/usr/bin/env python3
"""从本地已下载的 images20-track zip 抽帧 + 划分 train/内验/eval 三份清单。

用法:
  python prepare_local.py --train-zip data/raw/bdd100k/zips/images20-track-train-1.zip \
                          --val-zip   data/raw/bdd100k/zips/images20-track-val-1.zip \
                          --n-train 40 --n-intval 10 --n-eval 10

输出:
  data/raw/bdd100k/frames/<seq>/*.jpg
  data/meta/split.json   {"train":[...], "intval":[...], "eval":[...]}
"""
import argparse
import collections
import json
import os
import zipfile

LABELS = "data/raw/bdd100k/box_track_labels_trainval.zip"
GT2 = {"pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}


def seq_stats(label_zip, seq, split):
    try:
        data = json.loads(label_zip.read(f"bdd100k/labels/box_track_20/{split}/{seq}.json"))
    except KeyError:
        return None
    c = collections.Counter()
    for fr in data:
        for lb in (fr.get("labels") or []):
            if lb["category"] in GT2:
                c[lb["category"]] += 1
    n = len(data)
    return {
        "frames": n,
        "veh": c["car"] + c["truck"] + c["bus"] + c["motorcycle"] + c["bicycle"],
        "per": c["pedestrian"] + c["rider"],
        "veh_pf": round((c["car"] + c["truck"] + c["bus"]) / max(n, 1), 2),
        "per_pf": round((c["pedestrian"] + c["rider"]) / max(n, 1), 2),
        "n_cls": sum(1 for k in GT2 if c[k] > 0),
        "boxes": sum(c.values()),
    }


def pick(cands, n, stats, key):
    """按 key 排序取前 n；cands 为剩余序列集合"""
    order = sorted([s for s in cands], key=lambda s: stats[s][key], reverse=True)
    return order[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-zip", default="data/raw/bdd100k/zips/images20-track-train-1.zip")
    ap.add_argument("--val-zip", default="data/raw/bdd100k/zips/images20-track-val-1.zip")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--frames-root", default="data/raw/bdd100k/frames")
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-intval", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--keep-eval", default="b1e9ee0e-67e26f2e")
    args = ap.parse_args()

    lab = zipfile.ZipFile(args.labels)
    tz = zipfile.ZipFile(args.train_zip)
    vz = zipfile.ZipFile(args.val_zip)

    def seqs_of(z):
        return sorted({n.split("/")[4] for n in z.namelist()
                       if n.endswith(".jpg") and n.count("/") >= 5})

    train_all, val_all = seqs_of(tz), seqs_of(vz)
    st = {}
    for s in train_all:
        r = seq_stats(lab, s, "train")
        if r:
            st[s] = r
    for s in val_all:
        r = seq_stats(lab, s, "val")
        if r:
            st[s] = r
    print(f"train-1 zip: {len(train_all)} 条序列(有标签 {sum(1 for s in train_all if s in st)})")
    print(f"val-1  zip: {len(val_all)} 条序列(有标签 {sum(1 for s in val_all if s in st)})")

    # ---- 评测集（全部来自 val，绝不用训练）----
    keep = [s for s in args.keep_eval.split(",") if s]
    rest = set(val_all) - set(keep)
    n_ped = args.n_eval // 2
    n_car = args.n_eval - len(keep) - n_ped
    ev = list(keep)
    p1 = pick(rest, n_car, st, "veh_pf"); rest -= set(p1); ev += p1
    p2 = pick(rest, n_ped, st, "per_pf"); ev += p2
    ev = list(dict.fromkeys([s for s in ev if s in st]))

    # ---- 训练 + 内部验证（全部来自 train）----
    rest_t = set(train_all) & set(st)
    train = pick(rest_t, args.n_train, st, "veh_pf")
    rest_t -= set(train)
    intval = pick(rest_t, args.n_intval, st, "veh_pf")

    split = {"train": train, "intval": intval, "eval": ev}
    os.makedirs("data/meta", exist_ok=True)
    json.dump(split, open("data/meta/split.json", "w"), indent=1)

    # ---- 抽帧 ----
    os.makedirs(args.frames_root, exist_ok=True)
    total = 0
    for group, z in (("train", tz), ("intval", tz), ("eval", vz)):
        names = z.namelist()
        for seq in split[group]:
            outdir = os.path.join(args.frames_root, seq)
            os.makedirs(outdir, exist_ok=True)
            members = [n for n in names if f"/{seq}/" in n and n.endswith(".jpg")]
            got = 0
            for n in members:
                dst = os.path.join(outdir, os.path.basename(n))
                if os.path.exists(dst) and os.path.getsize(dst) > 1024:
                    got += 1
                    continue
                with open(dst, "wb") as f:
                    f.write(z.read(n))
                got += 1
            total += got
            print(f"  [{group}] {seq}  {got}/{len(members)} 帧")
    print(f"\n合计抽取 {total} 帧")
    print(f"划分: train {len(train)} / intval {len(intval)} / eval {len(ev)}  -> data/meta/split.json")


if __name__ == "__main__":
    main()
