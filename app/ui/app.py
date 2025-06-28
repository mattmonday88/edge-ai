import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
from ingest.pdf_reader import load_images_from_pdf
from layout.layout_detection_onnx import detect_layout
from ocr.trocr_onnx import extract_text_from_regions
from ner.ner_onnx import extract_entities
from summarizer.summarizer_onnx import generate_summary
from ocr.trocr_onnx import extract_text_from_images

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../uploads")
ANNOTATED_FOLDER = os.path.join(os.path.dirname(__file__), "../annotated")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        images = load_images_from_pdf(filepath)
        annotated_paths = []

        # LAYOUT DETECTION
        # for i, image in enumerate(images):
        #     boxes, scores, class_ids = detect_layout(image)
        #     annotated_image = image.copy()
        #     draw = ImageDraw.Draw(annotated_image)
        #     for box in boxes:
        #         x0, y0, x1, y1 = box
        #         x0, x1 = sorted([x0, x1])
        #         y0, y1 = sorted([y0, y1])
        #         draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                
        #     text = extract_text_from_regions(image, boxes)
        #     entities = extract_entities(text)
        #     summary = generate_summary(text)

        #     output_path = os.path.join(ANNOTATED_FOLDER, f"annotated_{i}.png")
        #     annotated_image.save(output_path)
        #     annotated_paths.append(f"/annotated/{os.path.basename(output_path)}")

                    # Load images from PDF
        images = load_images_from_pdf(filepath)

        # Full-page OCR
        extracted_texts = extract_text_from_images(images)
        full_text = " ".join(extracted_texts)

        # Summarization
        summary = generate_summary(full_text)

        # Named Entity Recognition
        entities = extract_entities(full_text)

        #return render_template("result.html", summary=summary, entities=entities)

        return render_template("result.html", image_paths=annotated_paths, summary=summary, entities=entities)
    
    return "No file uploaded", 400

@app.route("/annotated/<filename>")
def annotated_file(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)