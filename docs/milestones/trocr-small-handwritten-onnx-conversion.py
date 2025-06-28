from transformers import VisionEncoderDecoderModel
import torch
import torch.nn as nn

class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, decoder_input_ids, encoder_hidden_states):
        return self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states
        ).logits

# Load model
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
decoder = model.decoder
decoder.eval()

# Wrap decoder to keep both inputs alive
wrapped_decoder = DecoderWrapper(decoder)

# Correct hidden size
hidden_dim = model.config.decoder.d_model  # For trocr-small-handwritten, this should be 384
encoder_seq_len = 197
max_length = 32

# Dummy inputs
decoder_input_ids = torch.ones((1, max_length), dtype=torch.long)
encoder_hidden_states = torch.ones((1, encoder_seq_len, hidden_dim), dtype=torch.float)

# Export
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

print("✅ Decoder exported with correct input shapes.")
