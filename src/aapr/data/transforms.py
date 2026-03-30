import torch


class ImageAugmentation:
    """Simple tensor-based image augmentations."""

    def __init__(
        self,
        apply_flip: bool = True,
        apply_color_jitter: bool = True,
        apply_cutout: bool = True,
        jitter_strength: float = 0.1,
    ):
        self.apply_flip = apply_flip
        self.apply_color_jitter = apply_color_jitter
        self.apply_cutout = apply_cutout
        self.jitter_strength = jitter_strength

    def random_flip(self, image: torch.Tensor) -> torch.Tensor:
        return image.flip(-1)

    def color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.jitter_strength
        return (image * scale).clamp(0.0, 1.0)

    def cutout(self, image: torch.Tensor, max_frac: float = 0.2) -> torch.Tensor:
        _, height, width = image.shape
        cut_h = max(1, int(height * max_frac * torch.rand(1).item()))
        cut_w = max(1, int(width * max_frac * torch.rand(1).item()))
        top = torch.randint(0, max(height - cut_h, 1), (1,)).item()
        left = torch.randint(0, max(width - cut_w, 1), (1,)).item()
        out = image.clone()
        out[:, top : top + cut_h, left : left + cut_w] = 0.0
        return out

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        out = image
        if self.apply_flip and torch.rand(1).item() < 0.5:
            out = self.random_flip(out)
        if self.apply_color_jitter and torch.rand(1).item() < 0.5:
            out = self.color_jitter(out)
        if self.apply_cutout and torch.rand(1).item() < 0.5:
            out = self.cutout(out)
        return out
