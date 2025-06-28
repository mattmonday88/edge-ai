import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, decoder_input_ids, encoder_hidden_states):
        output = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states
        )
        return output.logits

# Load model
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
decoder = model.decoder
decoder.eval()

# Wrap decoder to preserve both inputs in ONNX
wrapped_decoder = DecoderWrapper(decoder)

# Get correct hidden size from decoder config
hidden_size = model.config.decoder.d_model  # Should be 384
encoder_seq_len = 197  # ViT typically outputs 197 tokens for 224x224 image
max_length = 32         # Max output token length

# Dummy input tensors with correct dimensions
decoder_input_ids = torch.ones((1, max_length), dtype=torch.long)
encoder_hidden_states = torch.ones((1, encoder_seq_len, hidden_size), dtype=torch.float)

# Export ONNX
torch.onnx.export(
    wrapped_decoder,
    args=(decoder_input_ids, encoder_hidden_states),
    f="trocr_decoder.onnx",
    input_names=["decoder_input_ids", "encoder_hidden_states"],
    output_names=["logits"],
    dynamic_axes={
        "decoder_input_ids": {0: "batch", 1: "sequence"},
        "encoder_hidden_states": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=14,
    do_constant_folding=True
)

print("✅ Decoder exported successfully with correct shapes.")

