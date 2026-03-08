import torch
import torch.nn as nn
from transformers import HubertModel, AutoProcessor


class SSLEmbeddingExtractor(nn.Module):
    """HuBERT/wav2vec2 SSL embeddings with learnable layer weighting."""

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        freeze_ssl: bool = True,
        num_layers_to_use: int = 13,  # HuBERT-base has 13 layers (1 CNN + 12 transformer)
    ):
        super().__init__()
        self.ssl_model = HubertModel.from_pretrained(model_name)
        self.ssl_model.config.output_hidden_states = True

        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False

        self.num_layers = num_layers_to_use
        self.layer_weights = nn.Parameter(torch.ones(num_layers_to_use) / num_layers_to_use)
        self.output_dim = self.ssl_model.config.hidden_size  # 768 for base

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, 1, T) raw audio
        Returns:
            features: (B, D, T') weighted sum of hidden states, transposed for Conv1D
        """
        x = waveform.squeeze(1)  # (B, T)
        outputs = self.ssl_model(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (B, T', D)

        # Weighted sum of layers
        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(hidden_states[: self.num_layers], dim=0)  # (L, B, T', D)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)  # (B, T', D)

        # Transpose to (B, D, T') for Conv1D processing
        return weighted.transpose(1, 2)
