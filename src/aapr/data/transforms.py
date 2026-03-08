import torch
import torchaudio


class AudioAugmentation:
    """Composable audio augmentations for training."""

    def __init__(
        self,
        noise_snr_db: float = 20.0,
        time_stretch_range: tuple[float, float] = (0.9, 1.1),
        pitch_shift_range: int = 2,
        apply_noise: bool = True,
        apply_time_mask: bool = True,
        sample_rate: int = 16000,
    ):
        self.noise_snr_db = noise_snr_db
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.apply_noise = apply_noise
        self.apply_time_mask = apply_time_mask
        self.sample_rate = sample_rate

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform)
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        snr_linear = 10 ** (self.noise_snr_db / 10)
        scale = (signal_power / (noise_power * snr_linear + 1e-8)).sqrt()
        return waveform + scale * noise

    def time_mask(self, waveform: torch.Tensor, max_mask_frac: float = 0.1) -> torch.Tensor:
        length = waveform.shape[-1]
        mask_len = int(length * max_mask_frac * torch.rand(1).item())
        start = torch.randint(0, max(length - mask_len, 1), (1,)).item()
        waveform = waveform.clone()
        waveform[..., start : start + mask_len] = 0
        return waveform

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.apply_noise and torch.rand(1).item() < 0.5:
            waveform = self.add_noise(waveform)
        if self.apply_time_mask and torch.rand(1).item() < 0.5:
            waveform = self.time_mask(waveform)
        return waveform
