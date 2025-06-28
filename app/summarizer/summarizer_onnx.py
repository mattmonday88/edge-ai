from transformers import AutoTokenizer
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("../models/distilbart-cnn-12-6.onnx")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128  # <-- FIXED length required by the ONNX model
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"]
    }

    outputs = session.run(None, ort_inputs)
    summary_ids = np.argmax(outputs[0], axis=-1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
