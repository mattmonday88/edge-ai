import onnxruntime as ort
import time
import psutil
import os
from transformers import AutoTokenizer, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIG ---
MODEL_ID = "microsoft/trocr-small-handwritten"
ENCODER_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\trocr-small-handwritten-onnx\trocr_encoder.onnx"
DECODER_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\trocr-small-handwritten-onnx\trocr_decoder.onnx"
IMAGE_PATH = "handwriting.jpg"
EXPECTED_TEXT = "hello world"  # Update this with your actual ground truth text
ITERATIONS = 20
MAX_LENGTH = 32
REPORT_PATH = "trocr_two_part_onnx_report.txt"
PLOT_PATH = "trocr_two_part_onnx_plot.png"

# --- LOAD TOKENIZER, PROCESSOR, IMAGE ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
processor = ViTImageProcessor.from_pretrained(MODEL_ID)
image = Image.open(IMAGE_PATH).convert("RGB")
pixel_values = processor(images=image, return_tensors="np").pixel_values

# --- LOAD ONNX SESSIONS ---
encoder_session = ort.InferenceSession(ENCODER_PATH, providers=["CPUExecutionProvider"])
decoder_session = ort.InferenceSession(DECODER_PATH, providers=["CPUExecutionProvider"])

print("\n[INFO] Decoder ONNX input names:")
for input in decoder_session.get_inputs():
    print(f"  - {input.name}")


# --- GREEDY DECODE FUNCTION ---
def greedy_decode(encoder_output, max_len=MAX_LENGTH):
    decoder_input_ids = np.array([[tokenizer.cls_token_id]], dtype=np.int64)
    for _ in range(max_len):
        logits = decoder_session.run(
            None,
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_output
            }
        )[0]
        next_token = np.argmax(logits[0, -1, :])
        if next_token == tokenizer.sep_token_id:
            break
        decoder_input_ids = np.append(decoder_input_ids, [[next_token]], axis=1)
    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

# --- BENCHMARKING LOOP ---
inference_times = []
memory_usages = []
correct = 0
decoded_outputs = []

for i in range(ITERATIONS):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024**2
    start_time = time.time()

    # Run encoder
    encoder_output = encoder_session.run(None, {"pixel_values": pixel_values})[0]

    # Decode output
    decoded_text = greedy_decode(encoder_output)
    decoded_outputs.append(decoded_text)

    end_time = time.time()
    end_mem = process.memory_info().rss / 1024**2

    # Record stats
    inference_times.append(end_time - start_time)
    memory_usages.append(end_mem - start_mem)
    if decoded_text.strip().lower() == EXPECTED_TEXT.lower():
        correct += 1

# --- METRICS SUMMARY ---
avg_time = np.mean(inference_times)
min_time = np.min(inference_times)
max_time = np.max(inference_times)
avg_mem = np.mean(memory_usages)
accuracy = (correct / ITERATIONS) * 100

# --- WRITE REPORT ---
report = f"""🧠 TrOCR ONNX Benchmark Report
Model: {MODEL_ID}
Iterations: {ITERATIONS}

🔹 Inference Time (s)
  Avg: {avg_time:.4f}
  Min: {min_time:.4f}
  Max: {max_time:.4f}

🔹 Memory Usage (MB)
  Avg: {avg_mem:.2f}

🎯 Accuracy (Exact Match): {accuracy:.2f}%
Target Text: "{EXPECTED_TEXT}"

Decoded Outputs:
{chr(10).join([f"{i+1:02d}: {text}" for i, text in enumerate(decoded_outputs)])}
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print(f"✅ Report saved to: {REPORT_PATH}")

# --- PLOT ---
plt.figure(figsize=(10, 5))
plt.plot(inference_times, label="Inference Time (s)")
plt.plot(memory_usages, label="Memory Usage (MB)")
plt.title("TrOCR ONNX Performance Over Time")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"📊 Plot saved to: {PLOT_PATH}")
