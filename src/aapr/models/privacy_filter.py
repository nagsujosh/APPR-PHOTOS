import torch
import torch.nn as nn


class VIBLayer(nn.Module):
    """Variational Information Bottleneck layer."""

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.mu_layer = nn.Conv1d(input_dim, bottleneck_dim, 1)
        self.logvar_layer = nn.Conv1d(input_dim, bottleneck_dim, 1)

    def forward(self, x: torch.Tensor, training: bool = True):
        """
        Args:
            x: (B, D, T)
        Returns:
            z: (B, bottleneck_dim, T) sampled representation
            kl_loss: scalar KL divergence
        """
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        # KL(q(z|x) || N(0,I)), mean over all elements (batch, dim, time)
        # Using .mean() keeps the scale comparable to utility loss regardless of T.
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl.mean()

        return z, kl_loss


class PrivacyFilter(nn.Module):
    """Conv1D stack + optional VIB bottleneck for privacy filtering."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        use_vib: bool = True,
        vib_beta: float = 1e-3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_vib = use_vib
        self.vib_beta = vib_beta

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.conv_stack = nn.Sequential(*layers)

        if use_vib:
            self.vib = VIBLayer(hidden_dim, output_dim)
        else:
            self.projection = nn.Conv1d(hidden_dim, output_dim, 1)

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, D_in, T)
        Returns:
            z: (B, D_out, T) filtered representation
            kl_loss: scalar (0 if VIB disabled)
        """
        h = self.conv_stack(x)

        if self.use_vib:
            z, kl_loss = self.vib(h, training=self.training)
            kl_loss = self.vib_beta * kl_loss
        else:
            z = self.projection(h)
            kl_loss = torch.tensor(0.0, device=x.device)

        return z, kl_loss
