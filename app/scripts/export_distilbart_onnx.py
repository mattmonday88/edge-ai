from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import onnx
import os

model_id = "sshleifer/distilbart-cnn-12-6"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()

dummy_input = tokenizer("This is a sample input.", return_tensors="pt", max_length=128, padding="max_length")
input_ids = dummy_input["input_ids"]

# Get root directory and model output path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(root_dir, "models")
os.makedirs(models_dir, exist_ok=True)

onnx_path = os.path.join(models_dir, "distilbart_cnn_12_6.onnx")

torch.onnx.export(
    model,
    (input_ids,),
    onnx_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14
)

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✔ distilbart_cnn_12_6.onnx exported and validated.")