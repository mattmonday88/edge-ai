
import os
import subprocess
import urllib.request
import shutil
import sys

# Define paths relative to the root of the app
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
scripts_dir = os.path.join(root_dir, "scripts", "yolov5")
weights_url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
weights_path = os.path.join(scripts_dir, "yolov5s.pt")
export_script = os.path.join(scripts_dir, "export.py")
onnx_output_path = os.path.join(scripts_dir, "yolov5s.onnx")
final_model_path = os.path.join(root_dir, "models", "yolov5s_placeholder.onnx")

# Ensure directories exist
os.makedirs(scripts_dir, exist_ok=True)
os.makedirs(os.path.join(root_dir, "models"), exist_ok=True)

# Step 1: Clone YOLOv5 if missing
if not os.path.exists(export_script):
    print("[INFO] Cloning YOLOv5 repository...")
    subprocess.run([
        "git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5", scripts_dir
    ], check=True)

# Step 2: Download yolov5s.pt if not already present
if not os.path.exists(weights_path):
    print("[INFO] Downloading yolov5s.pt...")
    urllib.request.urlretrieve(weights_url, weights_path)

# Step 3: Install pandas (required by YOLOv5 export)
try:
    import pandas
except ImportError:
    print("[INFO] Installing pandas...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=True)

# Step 4: Export to ONNX
print("[INFO] Exporting YOLOv5s to ONNX format...")
subprocess.run([
    sys.executable, "export.py",
    "--weights", "yolov5s.pt",
    "--imgsz", "640",
    "--batch", "1",
    "--include", "onnx",
    "--opset", "14"
], cwd=scripts_dir, check=True)

# Step 5: Move ONNX to models folder
if os.path.exists(onnx_output_path):
    shutil.move(onnx_output_path, final_model_path)
    print(f"[SUCCESS] Export complete: {final_model_path}")
else:
    print("[ERROR] Export failed, ONNX file not found.")
