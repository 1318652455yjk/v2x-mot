#!/usr/bin/env python3
"""批量抽取 BDD100K 跟踪序列帧（按分片分组，每个 zip 只读一次中央目录）。

输入：一个计划文件，每行 `seq,shard`（shard 形如 train-3），或由 --from-selection 自动生成。
用法：
  python fetch_batch.py --plan data/meta/fetch_plan_train10.csv
  python fetch_batch.py --from-selection --n 10 --out-plan data/meta/fetch_plan_train10.csv
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import json
import os
import time

from remotezip import RemoteZip

BASE = "https://huggingface.co/datasets/mrhwl123/BDD100kMOT/resolve/main/"
FRAME_DIR = "data/raw/bdd100k/frames"


def build_plan(n, sel_csv, shard_json, out_plan, exclude, skip=0):
    """从筛选结果里按桶均匀取 n 条，并附上分片号"""
    shards = json.load(open(shard_json))
    rows = [l.split(",") for l in open(sel_csv, encoding="utf-8").read().splitlines()[1:]
            if l and not l.startswith("#")]
    hdr = open(sel_csv, encoding="utf-8").readline().strip().split(",")
    b_i, s_i = hdr.index("bucket"), hdr.index("seq")
    by_bucket = collections.defaultdict(list)
    for r in rows:
        if r[s_i] in exclude:
            continue
        by_bucket[r[b_i]].append(r[s_i])
    base = n // len(by_bucket)
    quota = {b: base for b in by_bucket}
    for b in list(by_bucket)[: n - base * len(by_bucket)]:
        quota[b] += 1
    picked = []
    for b, q in quota.items():
        picked += by_bucket.get(b, [])[skip:skip + q]
    for b in by_bucket:                      # 不足则从其它桶补
        for s in by_bucket[b]:
            if len(picked) >= n:
                break
            if s not in picked:
                picked.append(s)
    picked = picked[:n]
    os.makedirs(os.path.dirname(out_plan), exist_ok=True)
    with open(out_plan, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq", "shard"])
        for s in picked:
            w.writerow([s, shards[s]])
    print(f"计划已生成: {out_plan}  ({len(picked)} 条)")
    return [(s, shards[s]) for s in picked]


def load_plan(path):
    with open(path, encoding="utf-8") as f:
        return [(r["seq"], r["shard"]) for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=None)
    ap.add_argument("--from-selection", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--skip", type=int, default=0, help="每个桶跳过前 skip 条（用于划分 train/内部验证）")
    ap.add_argument("--sel-csv", default="data/meta/bdd_seq_selection.csv")
    ap.add_argument("--shard-json", default="data/meta/train_shards.json")
    ap.add_argument("--out-plan", default="data/meta/fetch_plan_train10.csv")
    ap.add_argument("--exclude", default="b1e9ee0e-67e26f2e")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=FRAME_DIR)
    args = ap.parse_args()

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if args.from_selection or not args.plan:
        plan = build_plan(args.n, args.sel_csv, args.shard_json, args.out_plan, exclude, args.skip)
    else:
        plan = load_plan(args.plan)
        print(f"读取计划: {args.plan} ({len(plan)} 条)")

    by_shard = collections.defaultdict(list)
    for seq, shard in plan:
        by_shard[shard].append(seq)

    t_all = time.time()
    total_files = total_bytes = 0
    for shard, seqs in sorted(by_shard.items()):
        url = BASE + f"images20-track-{shard}.zip"
        t0 = time.time()
        print(f"\n=== {shard}: {len(seqs)} 条序列 ===")
        opened = False
        for attempt in (1, 2, 3):
            try:
                z = RemoteZip(url)
                names = z.namelist()
                opened = True
                break
            except Exception as e:
                print(f"  打开失败({type(e).__name__})，重试 {attempt}/3")
                time.sleep(3)
        if not opened:
            print(f"  !! {shard} 打开失败，跳过")
            continue
        members = {}
        for n in names:
            if not n.endswith(".jpg") or n.count("/") < 5:
                continue
            seq = n.split("/")[4]
            if seq in seqs:
                members.setdefault(seq, []).append(n)

        def work(item):
            seq, names_seq = item
            outdir = os.path.join(args.out, seq)
            os.makedirs(outdir, exist_ok=True)
            got = 0
            for name in sorted(names_seq):
                dst = os.path.join(outdir, os.path.basename(name))
                if os.path.exists(dst) and os.path.getsize(dst) > 1024:
                    got += 1
                    continue
                for _ in range(3):
                    try:
                        with open(dst, "wb") as f:
                            f.write(z.read(name))
                        got += 1
                        break
                    except Exception:
                        time.sleep(1)
            return seq, got, len(names_seq)

        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for seq, got, want in ex.map(work, members.items()):
                outdir = os.path.join(args.out, seq)
                sz = sum(os.path.getsize(os.path.join(outdir, f))
                         for f in os.listdir(outdir) if f.endswith(".jpg"))
                total_files += got
                total_bytes += sz
                flag = "OK " if got == want else "缺帧"
                print(f"  {flag} {seq}  {got}/{want} 帧  {sz/1e6:.1f} MB")
        z.close()
        print(f"  {shard} 用时 {time.time()-t0:.0f}s")

    print(f"\n总计: {total_files} 帧 / {total_bytes/1e6:.1f} MB / {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
