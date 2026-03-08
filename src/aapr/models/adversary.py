import torch
import torch.nn as nn

from .gradient_reversal import GradientReversalLayer


class MultiHeadAdversary(nn.Module):
    """Multi-head adversary with shared trunk and per-attribute heads.

    Supports gender, speaker_id, and age classification heads.
    Includes GRL and reset_parameters for adversary refresh.
    """

    def __init__(
        self,
        input_dim: int = 128,
        trunk_dim: int = 128,
        heads: dict[str, int] | None = None,
        dropout: float = 0.3,
        grl_lambda: float = 1.0,
    ):
        """
        Args:
            input_dim: dimension of filtered representation
            trunk_dim: hidden dimension for shared trunk
            heads: dict mapping attribute name to num_classes
                   e.g. {"gender": 2, "speaker_id": 91}
        """
        super().__init__()
        if heads is None:
            heads = {"gender": 2, "speaker_id": 91}

        self.grl = GradientReversalLayer(grl_lambda)

        # Temporal pooling (mean)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, trunk_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_dim, num_classes)
            for name, num_classes in heads.items()
        })

    def set_lambda(self, lambda_: float):
        self.grl.set_lambda(lambda_)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: (B, D, T) filtered representation
        Returns:
            dict of attribute_name -> (B, num_classes) logits
        """
        z_rev = self.grl(z)
        # Mean temporal pooling
        z_pooled = z_rev.mean(dim=2)  # (B, D)
        trunk_out = self.trunk(z_pooled)

        return {name: head(trunk_out) for name, head in self.heads.items()}

    def reset_parameters(self):
        """Re-initialize all parameters for adversary refresh."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
