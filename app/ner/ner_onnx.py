import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

session = ort.InferenceSession("../models/bert_base_ner.onnx")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

def extract_entities(text):
    inputs = tokenizer(text, return_tensors="np", truncation=True)
    # Only pass supported inputs
    available_inputs = session.get_inputs()
    input_names = [i.name for i in available_inputs]
    onnx_inputs = {k: v for k, v in inputs.items() if k in input_names}

    outputs = session.run(None, onnx_inputs)[0]
    predictions = np.argmax(outputs, axis=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return list(zip(tokens, predictions))
