#!/usr/bin/env python3
"""BDD100K box_track_20 标签 → YOLO 检测格式（无需联网）

输出结构（Ultralytics 可直接训练）:
  data/yolo/
    images/train/<seq>__<frame>.jpg   ← 软链接到原帧（不复制，省磁盘）
    images/val/<seq>__<frame>.jpg
    labels/train/<seq>__<frame>.txt   ← class cx cy w h（归一化）
    labels/val/<seq>__<frame>.txt
    data.yaml                          ← nc=8, names=[...]

用法:
  python bdd_to_yolo.py --seqs-file data/meta/plan_train30.csv --split train
  python bdd_to_yolo.py --seq b1e9ee0e-67e26f2e --split val
"""
import argparse
import csv
import json
import os
import zipfile

# 8 类（2020 跟踪版），顺序固定
CLASSES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
DROP = {"other person", "other vehicle", "trailer"}   # 与 COCO 不对应，丢弃


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/bdd100k/box_track_labels_trainval.zip")
    ap.add_argument("--frames-root", default="data/raw/bdd100k/frames")
    ap.add_argument("--out-root", default="data/yolo")
    ap.add_argument("--seqs-file", default=None, help="csv，含 seq 列")
    ap.add_argument("--seq", default=None, help="单条序列")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--link", action="store_true", default=True, help="用软链接代替复制")
    args = ap.parse_args()

    seqs = []
    if args.seqs_file:
        with open(args.seqs_file, encoding="utf-8") as f:
            seqs += [r["seq"] for r in csv.DictReader(f)]
    if args.seq:
        seqs.append(args.seq)
    seqs = [s for s in dict.fromkeys(seqs) if s]
    if not seqs:
        raise SystemExit("没有指定序列")

    img_dir = os.path.join(args.out_root, "images", args.split)
    lbl_dir = os.path.join(args.out_root, "labels", args.split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    stats = {"seq": 0, "img": 0, "box": 0, "drop": 0, "missing_img": 0}
    for seq in seqs:
        data = None
        for sp in ("train", "val"):
            try:
                data = json.loads(z.read(f"bdd100k/labels/box_track_20/{sp}/{seq}.json"))
                break
            except KeyError:
                continue
        if data is None:
            print(f"  [跳过] 标签里没有 {seq}")
            continue
        src_dir = os.path.join(args.frames_root, seq)
        n_box = 0
        for fr in data:
            name = fr["name"]
            src = os.path.join(src_dir, name)
            if not os.path.exists(src) or os.path.getsize(src) < 1024:
                stats["missing_img"] += 1
                continue
            stem = f"{seq}__{os.path.splitext(name)[0]}"
            dst = os.path.join(img_dir, stem + ".jpg")
            if not os.path.exists(dst):
                if args.link:
                    os.symlink(os.path.abspath(src), dst)
                else:
                    with open(src, "rb") as fi, open(dst, "wb") as fo:
                        fo.write(fi.read())
            lines = []
            for lb in (fr.get("labels") or []):
                cat = lb["category"]
                if cat in DROP:
                    stats["drop"] += 1
                    continue
                b = lb["box2d"]
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
                # 1) 裁剪到图内 2) 丢弃无效框
                x1, x2 = max(0.0, min(x1, 1280)), max(0.0, min(x2, 1280))
                y1, y2 = max(0.0, min(y1, 720)), max(0.0, min(y2, 720))
                w, h = x2 - x1, y2 - y1
                if w < 2 or h < 2:
                    continue
                cx, cy = (x1 + x2) / 2 / 1280, (y1 + y2) / 2 / 720
                lines.append(f"{CLASSES.index(cat)} {cx:.6f} {cy:.6f} {w/1280:.6f} {h/720:.6f}")
                n_box += 1
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            stats["img"] += 1
            stats["box"] += n_box - (n_box if False else 0)
        stats["seq"] += 1
        print(f"  {seq}: 生成 {len(data)} 帧标签，{n_box} 框")

    yaml_path = os.path.join(args.out_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out_root)}\n")
        f.write(f"train: images/train\nval: images/val\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES, ensure_ascii=False)}\n")
    print(f"\n汇总: 序列 {stats['seq']} / 帧 {stats['img']} / 丢框 {stats['drop']} / 缺帧 {stats['missing_img']}")
    print(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
