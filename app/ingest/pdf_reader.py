import fitz  # PyMuPDF
from PIL import Image
import io

def load_images_from_pdf(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images
