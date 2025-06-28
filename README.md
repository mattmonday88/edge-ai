# Edge AI Document Intelligence System

A portable, offline-ready AI system that performs full-page OCR, summarization, and named entity recognition (NER) on uploaded PDF documents. Built with ONNX-optimized models for deployment on edge devices.

---

## 📌 Features

- 📄 **Full-page OCR** using ONNX-exported `trocr-small-printed`
- ✂️ **Automatic text extraction** from each page of a PDF (no layout detection needed, layout detection didnt work)
- 📚 **Summarization** using `distilbart-cnn-12-6.onnx`
- 🔍 **Named Entity Recognition (NER)** using `bert-base-ner.onnx`
- ⚡️ **Edge-optimized inference** with ONNX Runtime (CPU)
- 🖼 Web-based UI built with Flask to upload PDFs and view results
- 🧠 Clean architecture: each model runs independently with modular preprocessing and ONNX inference logic

---

## 🚀 Architecture Overview

```text
                +----------------+
                |     Upload     |
                |   PDF File     |
                +-------+--------+
                        |
                        v
             +----------------------+
             |   Convert PDF pages  |
             |     to Images (PIL)  |
             +----------------------+
                        |
                        v
             +----------------------+
             |    OCR with TrOCR    |
             | (trocr-small-printed)|
             +----------------------+
                        |
                        v
         +--------------------------+      
         | Extracted Plain Text     |
         |                          |
         | + Summarization (DistilBART)      
         | + NER (BERT base NER)    |
         +--------------------------+
                        |
                        v
               +------------------+
               |  Output Results  |
               |  (HTML Template) |
               +------------------+
```

---

## 🧠 Models Used

| Component        | Model                          | Format | Size | Source |
|------------------|--------------------------------|--------|------|--------|
| OCR              | `trocr-small-printed`          | ONNX   | ~75MB| [Hugging Face](https://huggingface.co/microsoft/trocr-small-printed) |
| Summarization    | `distilbart-cnn-12-6`          | ONNX   | ~300MB | [Hugging Face](https://huggingface.co/sshleifer/distilbart-cnn-12-6) |
| Named Entity Rec.| `bert-base-NER`                | ONNX   | ~400MB | [Hugging Face](https://huggingface.co/dslim/bert-base-NER) |

> All models are converted to ONNX ahead of time and stored locally in the `models/` directory.

---

## 🛠 Requirements

- Python 3.10+
- pip or [uv](https://github.com/astral-sh/uv) (recommended for performance)
- ONNX Runtime
- Flask
- PyMuPDF
- Pillow
- Transformers (only for tokenizer utilities)

---

## 🧪 Running the App

```bash
# Clone the repo
git clone https://github.com/<your-org-or-username>/edge-ai-doc-intelligence.git
cd edge-ai-doc-intelligence

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app/ui/app.py
```

Open your browser to: `http://localhost:5000`

---

## 📁 Project Structure

```
app/
├── models/
│   ├── trocr-small-printed.onnx
│   ├── distilbart-cnn-12-6.onnx
│   └── bert-base-ner.onnx
├── ocr/
│   └── trocr_onnx.py
├── ner/
│   └── ner_onnx.py
├── summarizer/
│   └── summarizer_onnx.py
├── ui/
│   ├── app.py
│   └── templates/
│       └── index.html
└── scripts/
```

---

## 🧹 TODO / Future Work

- [ ] Add lightweight layout detection using YOLOv5 for PubLayNet
- [ ] Improve OCR filtering (remove headers, logos)
- [ ] Add PDF download of results
- [ ] Integrate TinyBERT for smaller NER pipeline
- [ ] Add Docker container for deployment

---

## 📜 License

This project is released under the MIT License.

---

## 🙋‍♂️ Maintainer

Built by [Matt Monday](https://github.com/your-username)  
For questions or support, please open an [issue](https://github.com/<your-repo>/issues).