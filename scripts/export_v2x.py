#!/usr/bin/env python3
"""把跟踪结果（jsonl）转成车联网运动目标状态消息（类 CAM / 感知目标列表）。

输出三样东西：
  <seq>_v2x.jsonl   每帧一条消息（含该帧所有目标）
  <seq>_v2x_sample.json  一条可读示例
  docs/V2X_SCHEMA.md     字段说明（首次运行自动生成）

坐标说明（重要）：
  BDD100K 是车载运动相机且**没有标定参数**，无法严格反投影到米制。
  因此这里给出三档位置：
    1) bbox_px / center_px     像素坐标（原始）
    2) center_norm             ［0,1］归一化（与分辨率无关）
    3) pos_m_est（可选）       基于"自车前方地面平面 + 假定焦距"的**粗略**米制估计，
                               仅用于演示消息结构，不作为精度结论（--assume-scale 开启）

用法:
  python export_v2x.py --jsonl out/b1e9ee0e-67e26f2e_final.jsonl --fps 5 --station RSU-001
  python export_v2x.py --jsonl ... --assume-scale 1108 --cam-height 1.4
"""
import argparse
import collections
import json
import math
import os

CLS_CODE = {"pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5,
            "train": 6, "motorcycle": 7, "bicycle": 8}


def heading_deg(vx, vy):
    """图像坐标系（y 向下）下的运动方向，转成"相对图像正上方"的角度，0=正上（远处），顺时针为正"""
    if vx == 0 and vy == 0:
        return None
    return round(math.degrees(math.atan2(vx, -vy)) % 360, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--station", default="VEH-EGO-001")
    ap.add_argument("--ttl", type=float, default=0.5, help="消息有效期(s)")
    ap.add_argument("--img-w", type=int, default=1280)
    ap.add_argument("--img-h", type=int, default=720)
    ap.add_argument("--assume-scale", type=float, default=None,
                    help="假定焦距(px)，给出米制粗估；1280x720 前视约 1100")
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = args.out or (os.path.splitext(args.jsonl)[0] + "_v2x")
    by_frame = collections.defaultdict(list)
    for line in open(args.jsonl, encoding="utf-8"):
        r = json.loads(line)
        by_frame[r["frame"]].append(r)

    n_msg = n_obj = 0
    sample = None
    with open(base + ".jsonl", "w", encoding="utf-8") as fo:
        for f in sorted(by_frame):
            ts = round(f / args.fps, 3)
            objs = []
            for r in sorted(by_frame[f], key=lambda x: -x["conf"]):
                x1, y1, x2, y2 = r["bbox"]
                cx, cy = r["center"]
                vx, vy = r["vel_px"]
                o = {
                    "track_id": r["id"],
                    "cls": r["cls"],
                    "cls_code": CLS_CODE.get(r["cls"], 0),
                    "bbox_px": [round(v, 1) for v in (x1, y1, x2, y2)],
                    "center_px": [round(cx, 1), round(cy, 1)],
                    "center_norm": [round(cx / args.img_w, 4), round(cy / args.img_h, 4)],
                    "size_px": [round(x2 - x1, 1), round(y2 - y1, 1)],
                    "vel_px_s": [round(vx * args.fps, 1), round(vy * args.fps, 1)],
                    "heading_deg": heading_deg(vx, vy),
                    "conf": r["conf"],
                    "state": r.get("state", "tracked"),
                }
                if args.assume_scale:
                    # 针孔模型 + 地面平面假设: Z = f*H/ (y_bottom - cy0), 取图像中心为光心
                    focal = args.assume_scale
                    yb = y2
                    if yb > args.img_h / 2 + 5:
                        Z = focal * args.cam_height / (yb - args.img_h / 2)
                        X = (cx - args.img_w / 2) * Z / focal
                        o["pos_m_est"] = [round(X, 2), round(Z, 2)]
                        o["speed_mps_est"] = round(
                            math.hypot(*(v * Z / focal for v in (vx * args.fps, vy * args.fps))), 2)
                        o["scale_source"] = "assumed_focal+ground_plane"
                objs.append(o)
            msg = {"station_id": args.station, "frame": f, "ts": ts,
                   "ttl": round(args.ttl, 2), "n_obj": len(objs), "objects": objs}
            fo.write(json.dumps(msg, ensure_ascii=False) + "\n")
            n_msg += 1
            n_obj += len(objs)
            if sample is None:
                sample = msg
    json.dump(sample, open(base + "_sample.json", "w"), ensure_ascii=False, indent=1)
    print(f"输出: {base}.jsonl  ({n_msg} 帧消息 / {n_obj} 个目标状态)")
    print(f"示例: {base}_sample.json")

    # schema 文档
    os.makedirs("docs", exist_ok=True)
    with open("docs/V2X_SCHEMA.md", "w", encoding="utf-8") as f:
        f.write("""# V2X 运动目标状态消息 schema（本项目实现版）

## 消息（每帧一条）
| 字段 | 类型 | 说明 |
|---|---|---|
| station_id | str | 感知站标识（车载 `VEH-EGO-001` / 路侧 `RSU-001`…） |
| frame | int | 源帧号（BDD 为 5 fps 序列帧号） |
| ts | float | 时间戳(秒) = frame / fps |
| ttl | float | 消息有效期(秒)，超过即视为过期目标 |
| n_obj | int | 本帧目标数 |
| objects | list | 目标状态列表 |

## 目标（objects[i]）
| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| track_id | int | — | 跟踪器维持的连续 ID（**ID 维持率的载体**） |
| cls / cls_code | str/int | — | 类别名与编码（1 ped, 2 rider, 3 car, 4 truck, 5 bus, 6 train, 7 moto, 8 bike） |
| bbox_px | float[4] | px | x1,y1,x2,y2 |
| center_px | float[2] | px | 框中心 |
| center_norm | float[2] | [0,1] | 归一化中心，跨分辨率可用 |
| size_px | float[2] | px | 宽、高 |
| vel_px_s | float[2] | px/s | 图像平面速度（已按 fps 换算） |
| heading_deg | float/null | deg | 运动方向，0=图像正上（远方），顺时针为正 |
| conf | float | — | 检测置信度 |
| state | str | — | tracked / lost |
| pos_m_est | float[2] | m | **可选**：假定焦距 + 地面平面下的粗略米制位置（X 右, Z 前），仅供演示 |
| speed_mps_est | float | m/s | **可选**：由米制位置推得的速度粗估 |

## 关键说明（写报告时必须保留）
- BDD100K 车载相机**无标定参数**，`pos_m_est` / `speed_mps_est` 基于假设（f=1108 px, 相机高 1.4 m, 地面平面），
  只能用于演示消息结构，**不能作为精度结论**。
- 若要真实米制轨迹，需改用带标定的数据集（KITTI Tracking 的 `calib` + 3D 真值）。
""")
    print("schema: docs/V2X_SCHEMA.md")


if __name__ == "__main__":
    main()
