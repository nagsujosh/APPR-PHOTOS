import torch
import torch.nn as nn

from .task_model import AttentionPooling


class TeacherModel(nn.Module):
    """Pre-trained teacher model for knowledge distillation.

    Trained directly on mel features without a privacy filter.
    Frozen during adversarial training; provides soft probability targets
    that guide the student (filter + task model) toward better utility.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = AttentionPooling(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, T) mel spectrogram features (unfiltered)
        Returns:
            logits: (B, num_classes)
        """
        pooled = self.pooling(features)
        return self.classifier(pooled)

    def freeze(self):
        """Freeze all parameters and set to eval mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
