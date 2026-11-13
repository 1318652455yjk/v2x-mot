#!/usr/bin/env python3
"""从 HuggingFace 上的 BDD100K images20-track zip 中，按需抽取指定序列的帧。

用法:
  python fetch_bdd_seq.py --seq b1e9ee0e-67e26f2e --split val
  python fetch_bdd_seq.py --seq 034a8bdf-f9e9ad8d --split train --shard 1

原理: 用 remotezip 通过 HTTP Range 只读取中央目录 + 目标序列的成员，
避免下载整个 5 GB 的 zip。ZIP64 由 Python 标准库 zipfile 处理。
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from remotezip import RemoteZip

REPO = "mrhwl123/BDD100kMOT"
FILES = {"val": "images20-track-val-1.zip", "train": "images20-track-train-{shard}.zip"}
URL = "https://huggingface.co/datasets/{repo}/resolve/main/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--shard", default=1, type=int)
    ap.add_argument("--out", default="data/raw/bdd100k/frames")
    ap.add_argument("--workers", default=8, type=int,
                    help="并发数；每个线程一个 RemoteZip 实例（会各自读一次中央目录）")
    args = ap.parse_args()

    url = URL.format(repo=REPO, name=FILES[args.split].format(shard=args.shard))
    outdir = os.path.join(args.out, args.seq)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    print(f"[1/3] 读远端目录: {url}")
    with RemoteZip(url) as z:
        names = [n for n in z.namelist() if n.endswith(".jpg")]
        seqs = sorted({n.split("/")[4] for n in names if n.count("/") >= 5})
        target = [n for n in names if f"/{args.seq}/" in n]
        if not target:
            print(f"[!] 分片内没有序列 {args.seq}；该分片共 {len(seqs)} 条序列，例如: {seqs[:5]}")
            sys.exit(1)
        target.sort()
        print(f"      分片序列数 {len(seqs)}，总帧 {len(names)}；目标字段 {len(target)} 帧")

        print(f"[2/3] 并发下载 {len(target)} 帧 ({args.workers} 线程)")
        # 按线程切分任务，每个线程独立 RemoteZip 实例
        chunks = [target[i::args.workers] for i in range(args.workers)]

        def worker(chunk):
            n_done = 0
            with RemoteZip(url) as zz:
                for n in chunk:
                    dst = os.path.join(outdir, os.path.basename(n))
                    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
                        n_done += 1
                        continue
                    with open(dst, "wb") as f:
                        f.write(zz.read(n))
                    n_done += 1
            return n_done

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            total = sum(ex.map(worker, chunks))

    files = [f for f in os.listdir(outdir) if f.endswith(".jpg")]
    size = sum(os.path.getsize(os.path.join(outdir, f)) for f in files)
    bad = [f for f in files if os.path.getsize(os.path.join(outdir, f)) < 1024]
    print(f"[3/3] 完成: {len(files)} 帧 / {size / 1e6:.1f} MB / 异常 {len(bad)} "
          f"/ 耗时 {time.time() - t0:.1f}s -> {outdir}")


if __name__ == "__main__":
    main()
