# Edge AI Development Environment

A comprehensive setup for Edge AI development with ONNX optimization, supporting both Windows and Linux/macOS systems.

## Overview

This repository provides the necessary tools and scripts to set up a complete Edge AI development environment. It's designed to work with the latest Python 3.12 and includes all dependencies needed for:

- Model optimization with ONNX
- Working with Hugging Face models
- Computer vision and NLP tasks
- Creating API endpoints for your models
- Building demo applications

## System Requirements

- Windows, macOS, or Linux
- Git Bash (recommended for Windows users)
- Internet connection for downloading packages
- At least 10GB of free disk space
- 8GB+ RAM recommended

## Quick Setup

### Option 1: Using the Automated Script

The included bash script will:
1. Install the UV package manager (a faster alternative to pip)
2. Create a Python 3.12 virtual environment
3. Install all required dependencies

```bash
# Make the script executable
chmod +x setup_edge_ai_env.sh

# Run the script
./setup_edge_ai_env.sh
```

### Option 2: Manual Setup

If you prefer to set up manually:

```bash
# Create a project directory
mkdir edge_ai_project
cd edge_ai_project

# Create a virtual environment (standard method)
python -m venv edge_ai_env

# Activate the environment
# On Windows:
source edge_ai_env/Scripts/activate
# On Linux/macOS:
# source edge_ai_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Setting Up Hugging Face Access

All models can be downloaded from Hugging Face. To access models:

1. **Create a Hugging Face account:**
   Visit https://huggingface.co/join

2. **Generate an access token:**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "edge-ai-project")
   - Select read permissions
   - Copy the token

3. **Login via the command line:**
   ```bash
   # Ensure your environment is activated
   # On Windows:
   source edge_ai_project/python3.12/Scripts/activate
   # On Linux/macOS:
   # source edge_ai_project/python3.12/bin/activate
   
   # Login to Hugging Face
   huggingface-cli login
   ```

## Working with Models

### Downloading Models from Hugging Face

```python
from huggingface_hub import snapshot_download

# Download a model
model_path = snapshot_download(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"Model downloaded to: {model_path}")
```

### Converting Models to ONNX

```python
from transformers import AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification

# Load and convert to ONNX
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
ort_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

# Save the ONNX model
ort_model.save_pretrained("./onnx_model")
```

### Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("./onnx_model/model.onnx")

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": input_data})
```

## Creating a Simple Demo

The environment includes Streamlit and Gradio for quickly creating web demos:

```python
# Example Streamlit app
import streamlit as st
from transformers import pipeline
from PIL import Image

# Load model
classifier = pipeline("image-classification")

st.title("Image Classifier Demo")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    results = classifier(image)
    
    st.write("## Results")
    for result in results:
        st.write(f"{result['label']}: {result['score']:.2%}")
```

## Troubleshooting

### Common Issues:

1. **Script Permission Errors:**
   ```bash
   chmod +x setup_edge_ai_env.sh
   ```

2. **Virtual Environment Activation Path:**
   - Windows: `source python3.12/Scripts/activate`
   - Linux/macOS: `source python3.12/bin/activate`

3. **Package Installation Errors:**
   If you encounter errors during installation:
   ```bash
   uv pip install --upgrade pip
   uv pip install -r requirements.txt --no-deps
   ```

4. **UV Installation Issues:**
   If UV installation fails, you can manually install it:
   ```bash
   # On Windows
   curl -LsSf https://astral.sh/uv/install.ps1 | pwsh
   
   # On Linux/macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Dependencies

The installed environment includes:

- **ONNX & Runtime**: For model optimization and deployment
- **Deep Learning Frameworks**: PyTorch, TensorFlow, JAX
- **Computer Vision**: OpenCV, PIL, scikit-image
- **NLP & Speech**: transformers, datasets, librosa
- **Hugging Face Tools**: huggingface_hub, accelerate
- **Web UIs**: streamlit, gradio
- **API Development**: FastAPI, uvicorn
- **Utilities**: numpy, pandas, matplotlib, etc.

## Additional Resources

- [ONNX Official Documentation](https://onnx.ai/get-started/)
- [Hugging Face Optimum Library](https://huggingface.co/docs/optimum/index)
- [Edge AI Benchmarking Tools](https://github.com/onnx/tensorflow-onnx/tree/master/benchmark)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [UV Package Manager](https://github.com/astral-sh/uv)