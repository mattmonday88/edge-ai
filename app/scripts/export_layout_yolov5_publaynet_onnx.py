import os
import sys
import urllib.request
import subprocess
from shutil import move

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
YOLO_ROOT = os.path.join(ROOT, "scripts", "yolov5")
MODEL_URL = "https://huggingface.co/anonymous-iccv1968/DocLayout_YOLO_PubLayNet_iccv1968/blob/main/publaynet_best.pt"
MODEL_PT = os.path.join(YOLO_ROOT, "yolov5s-publaynet.pt")
EXPORT_SCRIPT = os.path.join(YOLO_ROOT, "export.py")
OUTPUT_ONNX = os.path.join(ROOT, "app", "models", "yolov5s_publaynet.onnx")

# Prepare directories
os.makedirs(YOLO_ROOT, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_ONNX), exist_ok=True)

# Download the model checkpoint
if not os.path.exists(MODEL_PT):
    print("🔽 Downloading YOLOv5s PubLayNet checkpoint...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PT)
else:
    print("✅ Checkpoint already exists.")

# Export to ONNX
print("📦 Exporting to ONNX...")
subprocess.run([
    sys.executable, EXPORT_SCRIPT,
    "--weights", MODEL_PT,
    "--imgsz", "640",
    "--batch", "1",
    "--include", "onnx",
    "--opset", "14"
], cwd=YOLO_ROOT, check=True)

exported = os.path.join(YOLO_ROOT, "yolov5s-publaynet.onnx")
if os.path.exists(exported):
    move(exported, OUTPUT_ONNX)
    print(f"✅ Model exported to {OUTPUT_ONNX}")
else:
    raise RuntimeError("❌ ONNX export failed—published file not found.")
