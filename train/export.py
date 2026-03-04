"""
export.py  —  导出 YOLOv8n-detect ONNX（post-NMS 格式）

输出格式：
  output0: [1, 300, 6]
    每行 = [x1, y1, x2, y2, score, cls_id]（letterbox 坐标）

导出两版：
  web/model/best_static.onnx   固定 640×640
  web/model/best_dynamic.onnx  动态 shape，供 Web ORT（index.html）使用

用法：
  python train/export.py
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import os

pt_path = "runs/detect/camera_drop_nano/weights/best.pt"
if not os.path.exists(pt_path):
    raise FileNotFoundError(f"找不到权重文件: {pt_path}")

onnx_src = Path(pt_path).with_suffix(".onnx")
models_dir = Path("web/model")
models_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(pt_path)

# ── 静态模型（固定 640×640）────────────────────────────────────
print("导出静态模型 (dynamic=False)...")
model.export(format="onnx", imgsz=640, simplify=True, opset=17, dynamic=False)
dst_static = models_dir / "best_static.onnx"
shutil.copy(str(onnx_src), str(dst_static))
print(f"  → {dst_static}")

# ── 动态模型（Web ORT，支持可变分辨率）──────────────────────────
print("导出动态模型 (dynamic=True)...")
model.export(format="onnx", imgsz=640, simplify=True, opset=17, dynamic=True)
dst_dynamic = models_dir / "best_dynamic.onnx"
shutil.copy(str(onnx_src), str(dst_dynamic))
print(f"  → {dst_dynamic}")

print("\n完成：")
print(f"  静态: {dst_static}")
print(f"  动态: {dst_dynamic}")
