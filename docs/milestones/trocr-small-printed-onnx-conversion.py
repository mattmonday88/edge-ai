import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from PIL import Image

MODEL_ID = "microsoft/trocr-small-printed"
ONNX_EXPORT_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\trocr-small-printed-onnx\trocr-small-printed.onnx"
DEVICE = "cpu"

# Load model and processor
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_ID)

# Load local handwriting image
image = Image.open("printed.jpg").convert("RGB")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

# Provide dummy decoder input (e.g., BOS token)
decoder_input_ids = torch.tensor([[0]], device=DEVICE)


# Export to ONNX
torch.onnx.export(
    model,
    args=(pixel_values, decoder_input_ids),
    f=ONNX_EXPORT_PATH,
    input_names=["pixel_values", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch"},
        "decoder_input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=14,
    do_constant_folding=True
)

print(f"✅ ONNX model exported to: {ONNX_EXPORT_PATH}")
