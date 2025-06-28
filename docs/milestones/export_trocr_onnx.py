from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

model_id = "microsoft/trocr-small-printed"
processor = TrOCRProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

model.eval()

dummy_input = processor(images=["dummy.png"], return_tensors="pt").pixel_values

torch.onnx.export(
    model,
    (dummy_input,),
    "trocr_small_printed.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14
)