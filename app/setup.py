import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")

def main():
    # Step 1: Install required packages
    print("Installing Python dependencies...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")

    # Step 2: Ensure required folders exist
    ensure_folder("uploads")
    ensure_folder("static/previews")
    ensure_folder("models")

    # Step 3: Provide ONNX export instructions
    print("\n✔ Setup complete!")
    print("📁 Place your exported ONNX models in the 'models/' folder:")
    print("   - trocr_small_printed.onnx")
    print("   - layoutlmv3_small.onnx")
    print("   - distilbart_cnn_12_6.onnx")

    print("\n🚀 To run the app:")
    print("   cd ui && python app.py")

if __name__ == "__main__":
    main()