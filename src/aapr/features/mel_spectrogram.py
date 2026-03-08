import torch
import torch.nn as nn
import torchaudio


class MelSpectrogramExtractor(nn.Module):
    """Log-mel spectrogram feature extractor."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.output_dim = n_mels

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, 1, T) raw audio
        Returns:
            features: (B, n_mels, T') log-mel spectrogram
        """
        # Remove channel dim for MelSpectrogram
        x = waveform.squeeze(1)  # (B, T)
        mel = self.mel_spec(x)  # (B, n_mels, T')
        log_mel = torch.log(mel + 1e-9)
        return log_mel
