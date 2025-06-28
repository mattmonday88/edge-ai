from ingest.pdf_reader import load_images_from_pdf
from layout.layout_detection_onnx import detect_layout
from ocr.trocr_onnx import extract_text_from_regions
from ner.ner_pipeline import extract_named_entities
from summarization.summarize_onnx import generate_summary
from search.elastic_indexer import index_document

PDF_PATH = "./documents/sample.pdf"
DOC_ID = "doc_001"

images = load_images_from_pdf(PDF_PATH)
layout_data = detect_layout(images[0])

ocr_results = extract_text_from_regions(images[0], layout_data)
extracted_text = "\n".join([r["text"] for r in ocr_results])

entities = extract_named_entities(extracted_text)
summary = generate_summary(extracted_text)

index_document(
    doc_id=DOC_ID,
    text=extracted_text,
    summary=summary,
    entities=entities,
    layout=layout_data
)
print("Document processed and indexed successfully.")