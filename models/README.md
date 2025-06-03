# University of South Alabama - Edge AI Models Used

The models which have been selected for the project exist within the models folder of this project. The models were selected by asking ChatGPT (4o) for reccomendations using the following prompts:

## Prompt 1

"what are the 10 smallest ai models on huggingface that do document based ocr"

Output:

| Model Name                       | Parameters | Key Features                                         | Notes                                                            |                                                                                                            |
| -------------------------------- | ---------- | ---------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| **1. PP-OCR**                    | \~3.5M     | Ultra-lightweight OCR system                         | Designed for efficiency with minimal computational resources.    |                                                                                                            |
| **2. VISTA-OCR**                 | \~150M     | Unified text detection and recognition               | Lightweight architecture for interactive OCR tasks.              |                                                                                                            |
| **3. SmolDocling**               | 256M       | Multimodal document understanding                    | Efficiently parses documents into structured formats.            |                                                                                                            |
| **4. TrOCR Small (Printed)**     | \~300M     | Transformer-based OCR for printed text               | Balances performance with smaller model size.                    |                                                                                                            |
| **5. TrOCR Small (Handwritten)** | \~300M     | Optimized for handwritten text recognition           | Tailored for handwriting OCR tasks.                              |                                                                                                            |
| **6. Donut Small**               | \~400M     | End-to-end document understanding without OCR engine | Processes documents directly, preserving layout.                 |                                                                                                            |
| **7. MGP-STR**                   | \~400M     | Scene text recognition using Vision Transformer      | Effective for recognizing text in natural scenes.                |                                                                                                            |
| **8. GOT-OCR2**                  | 580M       | General OCR model handling various formats           | Supports plain and formatted text extraction.                    |                                                                                                            |
| **9. DeepSeek-VL2 Small**        | 2.8B       | Vision-language model for OCR and more               | Handles OCR, document understanding, and visual QA.              |                                                                                                            |
| **10. LayoutLMv2**               | 200M–400M  | Integrates text, layout, and image information       | Suitable for document classification and information extraction. | 

* Ultra-Lightweight Models: PP-OCR and VISTA-OCR are ideal for deployment on devices with strict resource constraints, such as microcontrollers or mobile devices.
* Balanced Performance: SmolDocling and TrOCR Small variants offer a good trade-off between model size and OCR accuracy, suitable for general document processing tasks.
* Advanced Capabilities: Donut Small and MGP-STR provide enhanced features like layout preservation and scene text recognition, respectively.
* Comprehensive Solutions: GOT-OCR2, DeepSeek-VL2 Small, and LayoutLMv2 cater to more complex OCR and document understanding needs, albeit with larger model sizes

Descriptions of the models can be found below.

### GOT-OCR-2.0-hf

https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf

# Promt 2 - Information extraction and Summarization

"what are the top 10 lightweight models on huggingface for information extraction and summarization"

| Model Name                  | Parameters | Task(s)                          | Notes                                        |                                                                                                                                                                          |
| --------------------------- | ---------- | -------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GLiNER-Multitask-Large-v0.5 | \~110M     | NER, Relation Extraction, etc.   | Versatile multitask extraction model         |                                                                                                                                                                          |
| NuExtract-Tiny              | \~500M     | Structured Data Extraction       | Converts text to JSON format                 |                                                                                                                                                                          |
| SmolDocling                 | \~256M     | OCR, Document Understanding      | Compact model for document parsing           |                                                                                                                                                                          |
| LiLT                        | Varies     | Multilingual Document Processing | Combines text and layout understanding       |                                                                                                                                                                          |
| LMDX                        | Varies     | Document Information Extraction  | State-of-the-art extraction capabilities     |                                                                                                                                                                          |
| T5-Small                    | \~60M      | Text Summarization               | General-purpose summarization model          |                                                                                                                                                                          |
| Flan-T5-Small               | \~80M      | Instruction-tuned Summarization  | Enhanced instruction-based summarization     |                                                                                                                                                                          |
| DistilBART                  | \~90M      | Abstractive Summarization        | Fast inference with minimal performance loss |                                                                                                                                                                          |
| Pegasus-Samsum              | \~223M     | Dialogue Summarization           | Specialized in conversational data           |                                                                                                                                                                          |
| BART-Base-CNN               | \~139M     | News Summarization               | Trained on CNN/DailyMail dataset             | ([Hugging Face][1], [Medium][2], [Hugging Face]


## distilbart-cnn-12-6

This model is going to be used for the text summariazation portion of the project.

https://huggingface.co/sshleifer/distilbart-cnn-12-6

Base: BART, distilled for speed and size

Pros: Retains ~95% of BART’s performance with ~40% fewer parameters

Use case: Good for extractive and abstractive summaries on CPUs or low-end GPUs

Model: sshleifer/distilbart-cnn-12-6 (for CNN/DailyMail-style summarization)