#!/usr/bin/env python3
"""并行分块下载 HuggingFace(hf-mirror) 大文件，支持断点续传。

为什么需要它：本机到 HF 单连接约 1.4 MB/s，8 并发约 3.7 MB/s；
hf-mirror 的 resolve 会 307 跳到 cas-bridge 签名 URL（1 小时过期），故每次重试都重新解析。

用法:
  python hf_download.py images20-track-train-1.zip images20-track-val-1.zip --out data/raw/bdd100k/zips
"""
import argparse
import concurrent.futures as cf
import os
import re
import sys
import time

import requests

MIRROR = "https://hf-mirror.com/datasets/{repo}/resolve/main/{name}"
REPO = "mrhwl123/BDD100kMOT"


def resolve(name, repo=REPO, retries=5):
    """返回 (直接可用的签名 URL, 文件大小)"""
    url = MIRROR.format(repo=repo, name=name)
    for i in range(retries):
        try:
            r = requests.get(url, allow_redirects=False, timeout=30)
            if r.status_code in (301, 302, 307, 308):
                loc = r.headers["location"]
                if loc.startswith("/"):
                    loc = "https://hf-mirror.com" + loc
                h = requests.head(loc, allow_redirects=True, timeout=30)
                size = int(h.headers.get("content-length", 0))
                if size == 0:  # HEAD 不给长度时退回 GET 探测
                    g = requests.get(loc, headers={"Range": "bytes=0-0"}, timeout=30)
                    m = re.search(r"/(\d+)$", g.headers.get("content-range", ""))
                    size = int(m.group(1)) if m else 0
                return loc, size
            if r.status_code == 200:
                size = int(requests.head(url, timeout=30).headers.get("content-length", 0))
                return url, size
            print(f"  解析返回 {r.status_code}，重试 {i+1}/{retries}")
        except Exception as e:
            print(f"  解析异常 {type(e).__name__}，重试 {i+1}/{retries}")
        time.sleep(3)
    raise RuntimeError(f"无法解析 {name}")


def grab(url, start, end, path, retries=6):
    """下载 [start, end] 到 path，支持续传"""
    want = end - start + 1
    for i in range(retries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= want:
            return want
        try:
            with requests.get(url, headers={"Range": f"bytes={start + have}-{end}"},
                              stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(path, "ab") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        except Exception as e:
            print(f"    分块重试 {i+1}/{retries}: {type(e).__name__}")
            time.sleep(2)
    return os.path.getsize(path) if os.path.exists(path) else 0


def download(name, out_dir, chunk_mb=64, workers=8):
    dst = os.path.join(out_dir, name)
    if os.path.exists(dst):
        print(f"[跳过] 已存在 {dst} ({os.path.getsize(dst)/1e9:.2f} GB)")
        return dst
    url, size = resolve(name)
    print(f"[开始] {name}  {size/1e9:.2f} GB  {size/(1<<20):.0f} MB  签名URL已获取")
    chunk = chunk_mb << 20
    spans = [(s, min(s + chunk - 1, size - 1)) for s in range(0, size, chunk)]
    parts = [f"{dst}.part{i:04d}" for i in range(len(spans))]
    t0 = time.time()
    done_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(grab, url, s, e, p): i for i, ((s, e), p) in enumerate(zip(spans, parts))}
        for fu in cf.as_completed(futs):
            done_bytes += fu.result()
            el = time.time() - t0
            print(f"  {done_bytes/1e9:.2f}/{size/1e9:.2f} GB  "
                  f"{done_bytes/1e6/max(el,1e-9):.2f} MB/s  {el/60:.1f} min", flush=True)
    # 拼接
    with open(dst, "wb") as fo:
        for p in parts:
            with open(p, "rb") as fi:
                while True:
                    b = fi.read(1 << 22)
                    if not b:
                        break
                    fo.write(b)
    for p in parts:
        os.remove(p)
    real = os.path.getsize(dst)
    ok = real == size
    print(f"[完成] {dst}  {real/1e9:.2f} GB  {'✅大小一致' if ok else '❌大小不符'}  "
          f"耗时 {(time.time()-t0)/60:.1f} min")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default="data/raw/bdd100k/zips")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for n in args.names:
        try:
            download(n, args.out, args.chunk_mb, args.workers)
        except Exception as e:
            print(f"[失败] {n}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
