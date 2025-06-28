from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch

model_id = "microsoft/layoutlmv3-small"
processor = LayoutLMv3Processor.from_pretrained(model_id)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_id)

model.eval()

dummy_input = {
    "input_ids": torch.randint(0, 30522, (1, 512)),
    "bbox": torch.randint(0, 1000, (1, 512, 4)),
    "attention_mask": torch.ones((1, 512), dtype=torch.long),
    "pixel_values": torch.rand((1, 3, 224, 224))
}

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["bbox"], dummy_input["attention_mask"], dummy_input["pixel_values"]),
    "layoutlmv3_small.onnx",
    input_names=["input_ids", "bbox", "attention_mask", "pixel_values"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14
)