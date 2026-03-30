import torch
import torch.nn as nn


class ImageCNNExtractor(nn.Module):
    """CNN image encoder that outputs sequence features (B, D, T)."""

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 128,
        hidden_dims: tuple[int, ...] = (32, 64, 96),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        current_in = in_channels
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Conv2d(current_in, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            current_in = hidden_dim

        layers.extend(
            [
                nn.Conv2d(current_in, output_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            ]
        )

        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) image tensor in [0, 1]
        Returns:
            features: (B, D, T) where T is flattened spatial locations
        """
        features = self.encoder(image)  # (B, D, H', W')
        return features.flatten(start_dim=2)  # (B, D, H'*W')
