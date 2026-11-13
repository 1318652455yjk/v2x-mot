#!/usr/bin/env python3
"""复现性校验：在固定输入上重跑关键环节，与"黄金基准"对比，输出 PASS/FAIL。

设计：三层校验
  L1 环境    torch/cuda/ultralytics/opencv 版本、GPU 可用
  L2 数据    序列数、帧数、坏帧数、权重文件 sha256
  L3 结果    跟踪+评测指标与 golden 对比（容差内视为一致）

用法:
  python verify_repro.py --make-golden     # 生成/更新基准（在有把握的机器上跑一次）
  python verify_repro.py                   # 校验（默认）
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

PY = sys.executable
GOLDEN = "repro/golden.json"
FIXED = {  # 固定校验用序列（不随 split.json 变化）
    "seq_vehicle": "b251064f-e7a165fd",
    "seq_pedestrian": "b1d22ed6-f1cac061",
}
TOL = {"IDF1": 0.02, "MOTA": 0.02, "id_switch_rate": 0.005, "IDSW_rel": 0.10, "recall": 0.02}


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"命令失败: {cmd}")
    return r.stdout


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def level1_env():
    out = {}
    import torch, ultralytics, cv2
    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    out["cuda_version"] = torch.version.cuda
    out["ultralytics"] = ultralytics.__version__
    out["opencv"] = cv2.__version__
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    checks = [
        ("torch 含 cu124", "cu124" in out["torch"]),
        ("CUDA 可用", out["cuda_available"]),
        ("ultralytics >= 8.3", tuple(map(int, out["ultralytics"].split(".")[:2])) >= (8, 3)),
        ("opencv 4.x", out["opencv"].startswith("4.")),
    ]
    return out, checks


def level2_data():
    meta = json.load(open("data/meta/split.json"))
    info = {}
    checks = []
    for g in ("train", "intval", "eval"):
        n_seq = len(meta[g])
        n_frames = 0
        bad = 0
        for s in meta[g]:
            fs = glob.glob(f"data/raw/bdd100k/frames/{s}/*.jpg")
            n_frames += len(fs)
            bad += sum(1 for f in fs if os.path.getsize(f) < 1024)
        info[g] = {"seqs": n_seq, "frames": n_frames, "bad_files": bad}
        checks.append((f"{g} 帧数>0", n_frames > 0))
        checks.append((f"{g} 无坏帧", bad == 0))
    for w in ("runs/detect/runs/bdd_ft2/weights/best.pt",):
        info[w] = sha256(w) if os.path.exists(w) else None
        checks.append((f"{os.path.basename(w)} 存在", info[w] is not None))
    return info, checks


def run_tracking(seq, model="yolo11s.pt", tag="verify"):
    outdir = "repro/out"
    os.makedirs(outdir, exist_ok=True)
    sh(f'{PY} scripts/track_video.py --source data/raw/bdd100k/frames/{seq} '
       f'--model {model} --fps 5 --imgsz 1280 --conf 0.05 --half '
       f'--tracker configs/track/botsort_cmc.yaml --tag {tag} --no-video --out {outdir}')
    sh(f'{PY} scripts/eval_tracks.py --jsonl {outdir}/{seq}_{tag}.jsonl --seq {seq} '
       f'--json-out {outdir}/{seq}_{tag}_eval.json')
    return json.load(open(f"{outdir}/{seq}_{tag}_eval.json"))


def level3_results(make_golden=False):
    res, checks = {}, []
    for name, seq in FIXED.items():
        r = run_tracking(seq)
        res[name] = {k: r[k] for k in ("IDF1", "MOTA", "IDSW", "id_switch_rate",
                                       "ID_keep_rate", "recall", "precision", "Frag")}
        print(f"  {name} ({seq}): IDF1={r['IDF1']:.4f} MOTA={r['MOTA']:.4f} "
              f"IDSW={r['IDSW']} 维持率={r['ID_keep_rate']:.4f}")
    if make_golden:
        json.dump(res, open(GOLDEN, "w"), indent=1)
        print(f"\n已写入基准: {GOLDEN}")
        return res, [("生成基准", True)]
    if not os.path.exists(GOLDEN):
        print(f"[!] 缺少基准 {GOLDEN}，先跑 --make-golden")
        return res, [("基准文件存在", False)]
    g = json.load(open(GOLDEN))
    for name, cur in res.items():
        ref = g.get(name)
        if not ref:
            checks.append((f"{name} 在基准中", False))
            continue
        for k in ("IDF1", "MOTA", "id_switch_rate", "recall"):
            ok = abs(cur[k] - ref[k]) <= TOL[k]
            checks.append((f"{name}.{k}: {cur[k]:.4f} vs 基准 {ref[k]:.4f} (±{TOL[k]})", ok))
        rel = abs(cur["IDSW"] - ref["IDSW"]) / max(ref["IDSW"], 1)
        checks.append((f"{name}.IDSW: {cur['IDSW']} vs 基准 {ref['IDSW']} (±{TOL['IDSW_rel']:.0%})",
                       rel <= TOL["IDSW_rel"]))
    return res, checks


def report(all_checks):
    print("\n" + "=" * 66)
    print(f"{'校验项':<62}{'结果'}")
    print("=" * 66)
    n_pass = 0
    for name, ok in all_checks:
        print(f"{name:<62}{'PASS' if ok else 'FAIL'}")
        n_pass += bool(ok)
    print("=" * 66)
    print(f"总计 {n_pass}/{len(all_checks)} 通过")
    return n_pass == len(all_checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-golden", action="store_true")
    ap.add_argument("--skip-results", action="store_true", help="只校验环境与数据（不跑 GPU 推理）")
    args = ap.parse_args()

    print("=== L1 环境 ===")
    env, c1 = level1_env()
    for k, v in env.items():
        print(f"  {k}: {v}")
    print("\n=== L2 数据 ===")
    info, c2 = level2_data()
    for g, v in info.items():
        if isinstance(v, dict):
            print(f"  {g}: {v['seqs']} 序列 / {v['frames']} 帧 / 坏帧 {v['bad_files']}")
    checks = c1 + c2
    if not args.skip_results:
        print("\n=== L3 结果（固定 2 条序列重跑）===")
        _, c3 = level3_results(args.make_golden)
        checks += c3
    ok = report(checks)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
