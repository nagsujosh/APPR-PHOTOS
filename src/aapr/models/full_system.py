import torch
import torch.nn as nn

from .privacy_filter import PrivacyFilter
from .task_model import TaskModel
from .adversary import MultiHeadAdversary


class FullSystem(nn.Module):
    """Composes feature extractor, privacy filter, task model, and adversary."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        privacy_filter: PrivacyFilter,
        task_model: TaskModel,
        adversary: MultiHeadAdversary,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.privacy_filter = privacy_filter
        self.task_model = task_model
        self.adversary = adversary

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W) input images
        Returns:
            dict with keys: 'utility_logits', 'privacy_logits', 'kl_loss', 'features', 'filtered'
        """
        features = self.feature_extractor(image)
        filtered, kl_loss = self.privacy_filter(features)
        utility_logits = self.task_model(filtered)
        privacy_logits = self.adversary(filtered)

        return {
            "utility_logits": utility_logits,
            "privacy_logits": privacy_logits,
            "kl_loss": kl_loss,
            "features": features.detach(),
            "filtered": filtered.detach(),
        }

    def forward_from_features(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass starting from pre-extracted features (skip feature extractor)."""
        filtered, kl_loss = self.privacy_filter(features)
        utility_logits = self.task_model(filtered)
        privacy_logits = self.adversary(filtered)

        return {
            "utility_logits": utility_logits,
            "privacy_logits": privacy_logits,
            "kl_loss": kl_loss,
            "features": features.detach(),
            "filtered": filtered.detach(),
        }
