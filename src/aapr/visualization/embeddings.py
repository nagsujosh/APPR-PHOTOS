import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def plot_embeddings(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    speaker_labels: np.ndarray,
    emotion_labels: np.ndarray,
    method: str = "tsne",
    save_path: str | None = None,
    max_samples: int = 2000,
):
    """Side-by-side embedding plots showing speaker cluster dissolution.

    Creates a 2x2 grid:
    - Top row: colored by speaker (before / after)
    - Bottom row: colored by emotion (before / after)
    """
    n = min(len(embeddings_before), max_samples)
    idx = np.random.choice(len(embeddings_before), n, replace=False)

    emb_b = embeddings_before[idx]
    emb_a = embeddings_after[idx]
    spk = speaker_labels[idx]
    emo = emotion_labels[idx]

    # Reduce dimensionality
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))

    proj_b = reducer.fit_transform(emb_b)

    if method == "umap" and HAS_UMAP:
        reducer2 = umap.UMAP(n_components=2, random_state=42)
    else:
        reducer2 = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
    proj_a = reducer2.fit_transform(emb_a)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top row: speaker coloring
    unique_speakers = np.unique(spk)
    cmap_spk = plt.cm.tab20(np.linspace(0, 1, min(len(unique_speakers), 20)))
    spk_colors = np.array([cmap_spk[s % 20] for s in spk])

    axes[0, 0].scatter(proj_b[:, 0], proj_b[:, 1], c=spk_colors, s=5, alpha=0.6)
    axes[0, 0].set_title("Before Filtering (by Speaker)")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    axes[0, 1].scatter(proj_a[:, 0], proj_a[:, 1], c=spk_colors, s=5, alpha=0.6)
    axes[0, 1].set_title("After Filtering (by Speaker)")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    # Bottom row: emotion coloring
    unique_emo = np.unique(emo)
    cmap_emo = plt.cm.Set1(np.linspace(0, 1, len(unique_emo)))
    emo_colors = np.array([cmap_emo[e % len(cmap_emo)] for e in emo])

    axes[1, 0].scatter(proj_b[:, 0], proj_b[:, 1], c=emo_colors, s=5, alpha=0.6)
    axes[1, 0].set_title("Before Filtering (by Emotion)")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    axes[1, 1].scatter(proj_a[:, 0], proj_a[:, 1], c=emo_colors, s=5, alpha=0.6)
    axes[1, 1].set_title("After Filtering (by Emotion)")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
