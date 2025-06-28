from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from torch import nn

# --- CONFIGURATION ---
MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
ONNX_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\GOT-OCR-2.0-ONNX\got_ocr2.onnx"

# --- Wrapper to ensure only logits tensor is exported ---
class GotOcrExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, pixel_values):
        # Forward pass: only return logits (first output)
        output = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            return_dict=False  # returns tuple: (logits, ...)
        )
        return output[0]  # logits

# --- Load processor and model from Hugging Face Hub ---
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()  # set model to evaluation mode

# --- Build dummy image input for export tracing ---
dummy_image = Image.new("RGB", (1024, 1024), color="white")
inputs = processor(dummy_image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # image tensor input

# --- Handle BOS and image patch token for decoder input ---
bos_token_id = processor.tokenizer.bos_token_id  # Beginning of sentence token

# Get the image patch token index from config (or tokenizer)
image_patch_token_id = getattr(model.config, "image_token_index", None)
if image_patch_token_id is None:
    # If not present, add <im_patch> token to tokenizer and resize model embeddings
    num_added = processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<im_patch>']})
    if num_added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
    image_patch_token_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")
    if image_patch_token_id is None or image_patch_token_id == processor.tokenizer.unk_token_id:
        raise ValueError("Unable to get image_patch_token_id.")

# --- GOT-OCR2 expects number of patch tokens to match vision encoder output ---
num_image_features = 256  # This value is from your config/error logs

# Build input_ids: [BOS] + [IM_PATCH] * num_image_features
input_ids = [bos_token_id] + [image_patch_token_id] * num_image_features
input_ids = torch.tensor([input_ids], dtype=torch.long)  # shape: (1, 1+num_image_features)

# --- Prepare the wrapper and run ONNX export ---
wrapper = GotOcrExportWrapper(model)
wrapper.eval()  # Ensure wrapper is in eval mode


with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (input_ids, pixel_values),  # Model inputs: (input_ids, pixel_values)
        ONNX_PATH,  # Output filename
        input_names=["input_ids", "pixel_values"],  # Input node names for ONNX
        output_names=["logits"],                   # Output node name for ONNX
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},         # Allow variable batch and seq length
            "pixel_values": {0: "batch", 2: "height", 3: "width"},  # Allow dynamic image size
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

print(f"✅ Exported {MODEL_ID} to {ONNX_PATH}")
