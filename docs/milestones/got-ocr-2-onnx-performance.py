import onnxruntime as ort
import numpy as np
import time
import psutil
import os
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# --- CONFIGURATION ---

ONNX_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\GOT-OCR-2.0-ONNX\got_ocr2.onnx"
MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
REPORT_PATH = "onnx_baseline_report.txt"
NUM_RUNS = 50  # Number of inferences for timing
DEVICE = "cpu"

# (Optional) Add your test set here as list of (image_path, ground_truth_text)
TEST_SET = [
    # ("test_image1.jpg", "Expected text output 1"),
    # ("test_image2.jpg", "Expected text output 2"),
]

# --- Load processor and model ---
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, trust_remote_code=True)

# --- Get bos_token_id ---
bos_token_id = processor.tokenizer.bos_token_id

# --- Get image_patch_token_id robustly ---
image_patch_token_id = getattr(model.config, "image_token_index", None)
if image_patch_token_id is None:
    # Check if <im_patch> is present in tokenizer, else add it
    if "<im_patch>" not in processor.tokenizer.get_vocab():
        num_added = processor.tokenizer.add_special_tokens({'additional_special_tokens': ['<im_patch>']})
        if num_added > 0:
            print("Added <im_patch> token to tokenizer.")
    image_patch_token_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")
    if image_patch_token_id is None or image_patch_token_id == processor.tokenizer.unk_token_id:
        raise ValueError("Unable to get <im_patch> token ID. Please check your model and tokenizer compatibility.")

# --- Set number of image features (patch tokens) ---
num_image_features = getattr(model.config, "image_feature_count", 256)  # fallback to 256

# --- Prepare dummy input for performance test ---
dummy_image = Image.new("RGB", (1024, 1024), color="white")
inputs = processor(dummy_image, return_tensors="pt")
pixel_values = inputs["pixel_values"].numpy()

# --- Prepare input_ids ---
input_ids = np.array([[bos_token_id] + [image_patch_token_id] * num_image_features], dtype=np.int64)

# --- ONNX Inference Session ---
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1  # Set higher if you want to test multi-threading
session = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])

# --- Inference Performance Test ---
print("Running inference benchmark...")
inference_times = []
process = psutil.Process()

# Warm up
session.run(None, {"input_ids": input_ids, "pixel_values": pixel_values})

for i in range(NUM_RUNS):
    start_mem = process.memory_info().rss
    start_time = time.perf_counter()
    outputs = session.run(None, {"input_ids": input_ids, "pixel_values": pixel_values})
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss
    inference_times.append((end_time - start_time, end_mem - start_mem))

# --- Calculate statistics ---
all_times = [t[0] for t in inference_times]
all_mems = [t[1] for t in inference_times]
avg_time = np.mean(all_times)
min_time = np.min(all_times)
max_time = np.max(all_times)
avg_mem = np.mean(all_mems)

model_size_mb = os.path.getsize(ONNX_PATH) / (1024*1024)

# --- Accuracy Test (optional, only if you have test set) ---
def decode_output(logits, processor):
    """Convert model logits to text using the processor's tokenizer."""
    # logits: (batch, seq_len, vocab_size)
    token_ids = np.argmax(logits, axis=-1)
    texts = [processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
    return texts

accuracy_results = []
if TEST_SET:
    print("Running accuracy test...")
    for img_path, gt_text in TEST_SET:
        image = Image.open(img_path).convert("RGB").resize((1024, 1024))
        inputs = processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].numpy()
        logits = session.run(None, {"input_ids": input_ids, "pixel_values": pixel_values})[0]
        pred_text = decode_output(logits, processor)[0]
        accuracy_results.append((img_path, gt_text, pred_text, gt_text.strip() == pred_text.strip()))

    accuracy = sum(x[-1] for x in accuracy_results) / len(accuracy_results) if accuracy_results else None
else:
    accuracy = None

# --- Write report ---
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("ONNX Baseline Performance Report\n")
    f.write("================================\n\n")
    f.write(f"Model: {ONNX_PATH}\n")
    f.write(f"Model Size: {model_size_mb:.2f} MB\n")
    f.write(f"Device: {DEVICE}\n\n")
    f.write("Inference Performance:\n")
    f.write(f"  Runs: {NUM_RUNS}\n")
    f.write(f"  Avg Inference Time: {avg_time*1000:.2f} ms\n")
    f.write(f"  Min Inference Time: {min_time*1000:.2f} ms\n")
    f.write(f"  Max Inference Time: {max_time*1000:.2f} ms\n")
    f.write(f"  Avg Memory Usage per Inference: {avg_mem/1024:.2f} KB\n\n")
    if accuracy is not None:
        f.write("Accuracy Test Results:\n")
        f.write(f"  Samples: {len(TEST_SET)}\n")
        f.write(f"  Accuracy: {accuracy*100:.2f}%\n\n")
        for img_path, gt_text, pred_text, match in accuracy_results:
            f.write(f"  - Image: {img_path}\n")
            f.write(f"    Ground Truth: {gt_text}\n")
            f.write(f"    Prediction  : {pred_text}\n")
            f.write(f"    Match: {'✅' if match else '❌'}\n\n")
    else:
        f.write("Accuracy Test: [Skipped, no test set provided]\n")

print(f"✅ Baseline performance report written to: {REPORT_PATH}")
