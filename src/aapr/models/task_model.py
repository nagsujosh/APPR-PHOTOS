import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Learnable temporal attention pooling."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T)
        Returns:
            pooled: (B, D)
        """
        x_t = x.transpose(1, 2)  # (B, T, D)
        weights = self.attention(x_t)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (x_t * weights).sum(dim=1)  # (B, D)
        return pooled


class TaskModel(nn.Module):
    """Utility classifier with attention pooling."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pooling = AttentionPooling(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D, T) filtered representation
        Returns:
            logits: (B, num_classes)
        """
        pooled = self.pooling(z)
        return self.classifier(pooled)
