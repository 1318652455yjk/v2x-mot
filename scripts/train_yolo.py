#!/usr/bin/env python3
"""在 BDD100K 8 类跟踪帧上微调 YOLO11（4 GB 显存友好）。

用法:
  python train_yolo.py                          # 默认 yolo11s, SGD lr0=0.005, 30 epoch
  python train_yolo.py --model yolo11n.pt --epochs 20 --batch 8
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="bdd_ft")
    ap.add_argument("--optimizer", default="SGD")
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--mosaic", type=float, default=0.8)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, amp=not args.no_amp, workers=args.workers, patience=args.patience,
        project="runs", name=args.name, exist_ok=True, pretrained=True,
        optimizer=args.optimizer, lr0=args.lr0, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=1.0, cos_lr=True, close_mosaic=8,
        mosaic=args.mosaic, scale=0.4, seed=0, plots=False, verbose=True,
    )
    print("训练完成，权重在 runs/detect/runs/%s/weights/" % args.name)


if __name__ == "__main__":
    main()
