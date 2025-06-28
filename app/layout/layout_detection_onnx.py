import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("../models/yolov5s_layout.onnx")

def detect_layout(image):
    image_resized = image.resize((640, 640))
    img_data = np.array(image_resized).astype(np.float32)
    img_data = img_data.transpose(2, 0, 1) / 255.0
    img_data = np.expand_dims(img_data, axis=0)
    outputs = session.run(None, {"images": img_data})[0]
    boxes, scores, class_ids = postprocess(outputs, image.size)
    return boxes, scores, class_ids

def postprocess(prediction, original_shape, conf_threshold=0.25):
    boxes = []
    scores = []
    class_ids = []
    for det in prediction[0]:
        if det[4] > conf_threshold:
            x1, y1, x2, y2 = det[0:4]
            boxes.append([x1 * original_shape[0] / 640, y1 * original_shape[1] / 640,
                          x2 * original_shape[0] / 640, y2 * original_shape[1] / 640])
            scores.append(float(det[4]))
            class_ids.append(int(det[5]))
    return boxes, scores, class_ids