import torch
import numpy as np
import onnx
import onnxruntime
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os
import time
import tracemalloc

# --------- PARAMETERS ---------
MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
ONNX_MODEL_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\GOT-OCR-2.0-ONNX\got_ocr2.onnx"
NUM_IMAGE_FEATURES = 256      # As required by the GOT-OCR2 vision encoder!
SEQ_LENGTH = NUM_IMAGE_FEATURES + 1
HEIGHT = 1024
WIDTH = 1024
NUM_WARMUP = 3
NUM_RUNS = 10



# --------- ONNX INFERENCE/BENCHMARK ---------
print("[4/5] Running ONNX inference benchmark...")
onnx_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# Prepare numpy inputs
np_input_ids = np.full((1, SEQ_LENGTH), image_patch_token_id, dtype=np.int64)
np_input_ids[:, 0] = bos_token_id
np_pixel_values = np.ones((1, 3, HEIGHT, WIDTH), dtype=np.float32)

inputs_feed = {
    "input_ids": np_input_ids,
    "pixel_values": np_pixel_values
}

# Warmup runs
for _ in range(NUM_WARMUP):
    _ = onnx_session.run(None, inputs_feed)

# Timing runs
tracemalloc.start()
times = []
for i in range(NUM_RUNS):
    t0 = time.perf_counter()
    _ = onnx_session.run(None, inputs_feed)
    t1 = time.perf_counter()
    times.append(t1 - t0)
tracemalloc.stop()

print("\n--- Benchmark Results ---")
print(f"ONNX Model: {ONNX_MODEL_PATH}")
print(f"input_ids shape: {np_input_ids.shape}, dtype: {np_input_ids.dtype}")
print(f"pixel_values shape: {np_pixel_values.shape}, dtype: {np_pixel_values.dtype}")
print(f"Runs: {NUM_RUNS} (after {NUM_WARMUP} warmup)")
print(f"Inference time avg: {np.mean(times):.4f} sec, min: {np.min(times):.4f}, max: {np.max(times):.4f}")
print("[Done]")
