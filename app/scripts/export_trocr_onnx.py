from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import onnx
import os
from PIL import Image
import numpy as np

# ----- Load model and processor -----
model_id = "microsoft/trocr-small-printed"
processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)
model.eval()

# ----- Dummy image input -----
# Generate a dummy tensor of the same shape as real image inputs (batch_size, 3, height, width)
# Typical input: (1, 3, 384, 384)
dummy_pixel_values = torch.randn(1, 3, 384, 384)

# Prepare decoder input ids with only the start token
decoder_start_token_id = model.config.decoder_start_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = processor.tokenizer.bos_token_id

decoder_input_ids = torch.tensor([[decoder_start_token_id]])

print("Decoder Start Token ID:", decoder_start_token_id)
print("Decoder Start Token:", processor.tokenizer.decode([decoder_start_token_id]))

# ----- Paths -----
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(root_dir, "models")
os.makedirs(models_dir, exist_ok=True)
onnx_path = os.path.join(models_dir, "trocr_small_printed.onnx")

# ----- Export -----
torch.onnx.export(
    model,
    (dummy_pixel_values, decoder_input_ids),
    onnx_path,
    input_names=["pixel_values", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "decoder_input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=14
)

# ----- Validate -----
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✔ trocr_small_printed.onnx exported and validated.")