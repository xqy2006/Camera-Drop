"""
训练 YOLOv8n-detect Camera-Drop 帧，完成后导出 ONNX。

用法：
  python train/train.py --data train/dataset/dataset.yaml

训练结果保存在 runs/detect/camera_drop/weights/best.pt
ONNX 输出格式（post-NMS）：
  output0: [1, 300, 6]  —— x1 y1 x2 y2 score cls_id
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset.yaml 路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0",
                    help="GPU 索引（如 '0'），无 GPU 时填 'cpu'")
    ap.add_argument("--name", default="camera_drop")
    ap.add_argument("--patience", type=int, default=20,
                    help="Early stopping 耐心轮次")
    ap.add_argument("--workers", type=int, default=8,
                    help="Dataloader 进程数")
    ap.add_argument("--cache", default="ram",
                    help="图像缓存：ram / disk / False")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")   # detect 模型，非 seg

    print(f"开始训练：epochs={args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
    results = model.train(
        data=args.data,
        task="detect",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        exist_ok=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成，最优模型：{best_pt}")


if __name__ == "__main__":
    main()
