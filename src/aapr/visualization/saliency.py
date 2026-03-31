from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_input_saliency(
    image: torch.Tensor,
    forward_fn,
    target_index: int,
    modules: list[torch.nn.Module],
    num_samples: int = 12,
    noise_std: float = 0.08,
    percentile_range: tuple[float, float] = (75.0, 99.5),
) -> np.ndarray:
    """Compute a stronger SmoothGrad-style saliency map for one image."""
    base = image.unsqueeze(0).clone().detach()
    saliency_sum = None

    for sample_idx in range(num_samples):
        noisy = base.clone()
        if sample_idx > 0 and noise_std > 0:
            noisy = (noisy + noise_std * torch.randn_like(noisy)).clamp(0.0, 1.0)
        noisy.requires_grad_(True)

        for module in modules:
            module.zero_grad(set_to_none=True)

        logits = forward_fn(noisy)
        score = logits[0, int(target_index)]
        score.backward()

        grad = noisy.grad.detach().abs().amax(dim=1).squeeze(0)
        saliency_sum = grad if saliency_sum is None else saliency_sum + grad

    saliency = saliency_sum / max(num_samples, 1)
    saliency = saliency.cpu().numpy()

    low, high = np.percentile(saliency, percentile_range)
    scale = max(high - low, 1e-8)
    saliency = np.clip((saliency - low) / scale, 0.0, 1.0)
    saliency = saliency ** 0.7
    return saliency


def overlay_saliency(
    image: torch.Tensor,
    saliency: np.ndarray,
    alpha: float = 0.75,
    cmap_name: str = "magma",
) -> np.ndarray:
    """Overlay a saliency map on top of an RGB image tensor."""
    base = image.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
    heat = cm.get_cmap(cmap_name)(saliency)[..., :3]
    mask = np.clip(saliency[..., None] ** 0.8, 0.0, 1.0)
    background = 0.28 * base + 0.05
    highlighted = (1.0 - alpha) * base + alpha * heat
    return np.clip(background * (1.0 - mask) + highlighted * mask, 0.0, 1.0)


def plot_saliency_grid(rows: list[dict], save_path: str | Path):
    """Save a grid comparing utility and privacy saliency on real images."""
    save_path = Path(save_path)
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["original"])
        axes[row_idx, 0].set_title(row["original_title"])
        axes[row_idx, 1].imshow(row["utility_overlay"])
        axes[row_idx, 1].set_title(row["utility_title"])
        axes[row_idx, 2].imshow(row["privacy_overlay"])
        axes[row_idx, 2].set_title(row["privacy_title"])

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
