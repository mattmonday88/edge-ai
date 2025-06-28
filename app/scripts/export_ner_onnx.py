from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import onnx
import os

# Load the model
model_id = "dslim/bert-base-NER"
model = AutoModelForTokenClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Setup ONNX export path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(root_dir, "models")
os.makedirs(models_dir, exist_ok=True)
onnx_path = os.path.join(models_dir, "bert_base_ner.onnx")

# Create dummy input
dummy = tokenizer("John Doe works at Nfina Technologies in Alabama.", return_tensors="pt", max_length=64, padding="max_length")

# Export the model
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"}
    },
    opset_version=14
)

# Validate ONNX model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✔ bert_base_ner.onnx exported and validated.")