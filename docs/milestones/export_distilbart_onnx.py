from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_id = "sshleifer/distilbart-cnn-12-6"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.eval()

dummy_input = tokenizer("This is a sample input.", return_tensors="pt", max_length=128, padding="max_length")
input_ids = dummy_input["input_ids"]

torch.onnx.export(
    model,
    (input_ids,),
    "distilbart_cnn_12_6.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14
)