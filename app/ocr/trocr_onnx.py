import onnxruntime as ort
from PIL import Image
from transformers import TrOCRProcessor
import numpy as np

# Load ONNX model and processor
session = ort.InferenceSession("../models/trocr_small_printed.onnx")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

def run_trocr_onnx(image):
    # Preprocess image
    inputs = processor(images=image, return_tensors="np")
    pixel_values = inputs["pixel_values"]

    # Prepare decoder input IDs (start token)
    decoder_input_ids = np.array([[processor.tokenizer.cls_token_id]], dtype=np.int64)

    # Run inference
    ort_inputs = {
        "pixel_values": pixel_values,
        "decoder_input_ids": decoder_input_ids
    }
    outputs = session.run(None, ort_inputs)[0]

    # Decode
    pred_ids = np.argmax(outputs, axis=-1)
    decoded_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    return decoded_text.strip()

# using regions from layout detection, NOT working due to poor layout detection
def extract_text_from_regions(image, boxes):
    texts = []
    for box in boxes:
        x0, y0, x1, y1 = box
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        cropped = image.crop((x0, y0, x1, y1))
        text = run_trocr_onnx(cropped)
        texts.append(text)
        print(f"[OCR] Region: {text}")
    return " ".join(texts)

# Swapped to full page OCR
def extract_text_from_images(images):
    extracted_texts = []
    for image in images:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.numpy()

        decoder_input_ids = np.array([[processor.tokenizer.cls_token_id]], dtype=np.int64)
        decoded_ids = []

        for _ in range(64):  # limit max output length
            ort_inputs = {
                "pixel_values": pixel_values,
                "decoder_input_ids": decoder_input_ids
            }
            logits = session.run(None, ort_inputs)[0]
            next_token_logits = logits[:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1)

            if next_token_id[0] == processor.tokenizer.eos_token_id:
                break

            decoded_ids.append(next_token_id[0])
            decoder_input_ids = np.concatenate(
                [decoder_input_ids, next_token_id.reshape(1, 1)], axis=-1
            )

        decoded_text = processor.tokenizer.decode(decoded_ids, skip_special_tokens=True)
        extracted_texts.append(decoded_text)

    return extracted_texts