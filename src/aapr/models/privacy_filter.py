import torch
import torch.nn as nn


class MultiScaleConvBlock(nn.Module):
    """Parallel convolutions at k=3/7/15 with split output channels.

    Each branch produces out_dim//3 channels; concatenated they form out_dim.
    No projection layer needed, keeping parameter count close to a single Conv1d.

    Captures phoneme-level (k=3), syllable-level (k=7), and
    prosodic-level (k=15) temporal patterns simultaneously.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        b1 = out_dim // 3
        b2 = out_dim // 3
        b3 = out_dim - b1 - b2        # absorbs remainder
        self.b3  = nn.Conv1d(in_dim, b1, kernel_size=3,  padding=1)
        self.b7  = nn.Conv1d(in_dim, b2, kernel_size=7,  padding=3)
        self.b15 = nn.Conv1d(in_dim, b3, kernel_size=15, padding=7)
        self.norm = nn.InstanceNorm1d(out_dim, affine=True)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        return self.drop(self.act(self.norm(out)))


class VIBLayer(nn.Module):
    """Variational Information Bottleneck layer."""

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.mu_layer     = nn.Conv1d(input_dim, bottleneck_dim, 1)
        self.logvar_layer = nn.Conv1d(input_dim, bottleneck_dim, 1)

    def forward(self, x: torch.Tensor, training: bool = True):
        """
        Args:
            x: (B, D, T)
        Returns:
            z: (B, bottleneck_dim, T)
            kl_loss: scalar
        """
        mu     = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        if training:
            std = torch.exp(0.5 * logvar)
            z   = mu + torch.randn_like(std) * std
        else:
            z = mu

        # KL(q(z|x) || N(0,I)), mean over all elements keeps scale independent of T
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return z, kl_loss


class PrivacyFilter(nn.Module):
    """Multi-scale Conv1D + optional VIB bottleneck for privacy filtering.

    Architecture:
      1. MultiScaleConvBlock  — captures k=3/7/15 timescales with InstanceNorm
         (InstanceNorm instead of BatchNorm avoids speaker-level stat leakage)
      2. Conv1d stack (num_layers-1) — deeper representation with InstanceNorm
      3. VIB / linear projection — information bottleneck

    Note: self-attention was intentionally omitted from the filter. It adds global
    temporal context which encodes more speaker identity information, giving the
    adversary more to exploit and causing utility collapse after adversary refresh.
    """

    def __init__(
        self,
        input_dim:  int   = 128,
        hidden_dim: int   = 256,
        output_dim: int   = 128,
        num_layers: int   = 3,
        use_vib:    bool  = True,
        vib_beta:   float = 1e-3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.use_vib  = use_vib
        self.vib_beta = vib_beta

        # Layer 1: multi-scale feature extraction
        self.ms_block = MultiScaleConvBlock(input_dim, hidden_dim, dropout)

        # Layers 2+: standard depth with InstanceNorm
        mid_layers = []
        for _ in range(num_layers - 1):
            mid_layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.InstanceNorm1d(hidden_dim, affine=True),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.conv_stack = nn.Sequential(*mid_layers)

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
            z: (B, D_out, T)
            kl_loss: scalar (0 if VIB disabled)
        """
        h = self.ms_block(x)
        h = self.conv_stack(h)

        if self.use_vib:
            z, kl_loss = self.vib(h, training=self.training)
            kl_loss = self.vib_beta * kl_loss
        else:
            z = self.projection(h)
            kl_loss = torch.tensor(0.0, device=x.device)

        return z, kl_loss
