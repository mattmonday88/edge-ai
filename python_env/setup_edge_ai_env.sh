#!/bin/bash
# Setup script for Edge AI Development Environment
# Installs UV, Python 3.12, and all dependencies with latest versions

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Setting up Edge AI Development Environment with UV and Python 3.12..."

# Check if running with sudo/root permissions
if [ "$EUID" -eq 0 ]; then
  echo "⚠️ Warning: This script is running with sudo/root permissions."
  echo "It's recommended to run without sudo to avoid permission issues with Python packages."
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Create directory for our project
PROJECT_DIR="edge_ai_project"
mkdir -p $PROJECT_DIR

# Change to project directory
cd $PROJECT_DIR

# Create requirements.txt directly in the project directory
echo "📝 Creating requirements.txt file..."

# Install UV if not already installed
if ! command -v uv &> /dev/null; then
  echo "🔧 Installing UV package manager..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Add UV to PATH for this session
  source $HOME/.local/bin/env
  
  # Verify UV installation
  uv --version
  echo "✅ UV installed successfully!"
else
  echo "✅ UV already installed, version: $(uv --version)"
fi

# Use UV to install Python 3.12
echo "🐍 Installing Python 3.12 using UV..."
mkdir -p python3.12
uv venv python3.12 --python 3.12

# Activate the virtual environment - Fixed for Windows
echo "🔌 Activating virtual environment..."
source python3.12/Scripts/activate

# Verify Python version
echo "📋 Verifying Python version..."
python --version
if ! python --version | grep -q "3.12"; then
  echo "❌ Failed to install Python 3.12. Please check UV setup and try again."
  exit 1
fi

# Verify requirements.txt exists
echo "📝 Creating requirements.txt file..."
cat > requirements.txt << 'EOF'
# Core ONNX packages
onnx
onnxruntime
optimum

# Deep Learning Frameworks
torch
torchvision
tensorflow
jax
flax

# Computer Vision Libraries
opencv-python-headless  # Headless version recommended for edge deployments
pillow
scikit-image

# NLP & Speech Processing
transformers
datasets
soundfile
librosa
sentencepiece

# Hugging Face Tools
huggingface_hub
accelerate

# Model Optimization Tools
neural-compressor

# Utilities & Data Processing
numpy
scipy
pandas
matplotlib
seaborn
jupyterlab
ipywidgets
tqdm

# Web UI (for demos)
streamlit
gradio

# API Development
fastapi
uvicorn[standard]

# Benchmarking & Profiling
psutil
py-spy
EOF

# Install dependencies using UV
echo "📦 Installing dependencies with UV (this may take a while)..."
uv pip install -r requirements.txt

# Verify key installations
echo "🔍 Verifying installations..."
python -c "import onnx, onnxruntime, torch, tensorflow, fastapi; print('ONNX:', onnx.__version__, '\nONNX Runtime:', onnxruntime.__version__, '\nPyTorch:', torch.__version__, '\nTensorFlow:', tensorflow.__version__, '\nFastAPI:', fastapi.__version__)"

echo ""
echo "🎉 Edge AI environment setup is complete!"
echo "📋 Summary:"
echo "  - UV package manager installed"
echo "  - Python 3.12 installed"
echo "  - All dependencies installed with latest versions"
echo "  - FastAPI included for API development"
echo ""
echo "🔄 To activate this environment in the future, run:"
echo "  source $PWD/python3.12/Scripts/activate"
echo ""
echo "🚀 Ready for Edge AI development!"