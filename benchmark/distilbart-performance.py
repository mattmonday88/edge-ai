import os
import time
import psutil
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import torch
import matplotlib.pyplot as plt

# --------- SETTINGS ---------
onnx_model_path = r"C:\Users\Administrator\Documents\edge-ai\models\distilbart-cnn-12-6-onnx\distilbart-cnn-12-6.onnx"
model_name = "sshleifer/distilbart-cnn-12-6"
sample_text = "This is a test sentence to summarize for performance measurement."
num_iterations = 100
report_file = "distilbart-onnx-performance_report.txt"
plot_file = "distilbart-onnx-performance_plot.png"

# --------- LOAD TOKENIZER AND PREPARE INPUT ---------
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    padding="max_length",
    max_length=64,
    truncation=True
)

onnx_inputs = {
    "input_ids": inputs["input_ids"].numpy(),
    "attention_mask": inputs["attention_mask"].numpy()
}

# --------- MODEL SIZE ---------
model_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)

# --------- INFERENCE SESSION ---------
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
process = psutil.Process(os.getpid())

# --------- WARM-UP INFERENCE ---------
for _ in range(2):
    session.run(None, onnx_inputs)

# --------- COLLECT PERFORMANCE DATA ---------
inference_times = []
memory_deltas = []
rss_memory = []
vms_memory = []

with open(report_file, "w") as f:
    f.write(f"ONNX model: {onnx_model_path}\n")
    f.write(f"Model size: {model_size_mb:.2f} MB\n")
    f.write(f"Number of iterations: {num_iterations}\n\n")
    f.write(f"{'Iteration':>10} {'Inference Time (ms)':>22} {'Mem Delta (MB)':>20} {'RSS (MB)':>15} {'VMS (MB)':>15}\n")

    for i in range(num_iterations):
        mem_before = process.memory_info()
        start_time = time.perf_counter()
        session.run(None, onnx_inputs)
        end_time = time.perf_counter()
        mem_after = process.memory_info()

        inference_time = (end_time - start_time) * 1000
        mem_delta = (mem_after.rss - mem_before.rss) / (1024 * 1024)
        rss = mem_after.rss / (1024 * 1024)
        vms = mem_after.vms / (1024 * 1024)

        inference_times.append(inference_time)
        memory_deltas.append(mem_delta)
        rss_memory.append(rss)
        vms_memory.append(vms)

        f.write(f"{i+1:>10} {inference_time:>22.2f} {mem_delta:>20.2f} {rss:>15.2f} {vms:>15.2f}\n")

    f.write("\n")
    f.write(f"{'Average inference time:':<30} {np.mean(inference_times):.2f} ms\n")
    f.write(f"{'Average memory delta:':<30} {np.mean(memory_deltas):.2f} MB\n")
    f.write(f"{'Average RSS memory:':<30} {np.mean(rss_memory):.2f} MB\n")
    f.write(f"{'Average VMS memory:':<30} {np.mean(vms_memory):.2f} MB\n")

# --------- PLOT PERFORMANCE ---------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(inference_times, label="Inference Time (ms)")
plt.ylabel("Time (ms)")
plt.title("Inference Time per Iteration")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(memory_deltas, label="Memory Delta (MB)", color='orange')
plt.ylabel("Delta (MB)")
plt.title("Memory Delta per Iteration")
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(rss_memory, label="RSS Memory (MB)", color='green')
plt.plot(vms_memory, label="VMS Memory (MB)", color='red')
plt.xlabel("Iteration")
plt.ylabel("Memory (MB)")
plt.title("RSS and VMS Memory Usage")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(plot_file)
print(f"Plot saved as {plot_file}")
print(f"Report saved as {report_file}")
