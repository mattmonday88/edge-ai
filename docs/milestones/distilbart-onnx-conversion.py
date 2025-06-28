import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Model name (can be local path if downloaded)
model_name = "sshleifer/distilbart-cnn-12-6"
ONNX_PATH = r"C:\Users\Administrator\Documents\edge-ai\models\distilbart-cnn-12-6-onnx\distilbart-cnn-12-6.onnx"

# 2. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Create example input for export
text = "This is a test sentence to summarize."
inputs = tokenizer(text, return_tensors="pt")

# 4. Define input and output names
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]

# 5. Set model to evaluation mode
model.eval()

# 6. Export to ONNX
torch.onnx.export(
    model, 
    (inputs["input_ids"], inputs["attention_mask"]),         # model inputs as tuple
    ONNX_PATH,                             # file name
    input_names=input_names,                                # input names
    output_names=output_names,                              # output names
    opset_version=17,                                       # ONNX opset
    do_constant_folding=True,                               # optimize
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    }
)

print("Export completed: distilbart-cnn-12-6.onnx")
