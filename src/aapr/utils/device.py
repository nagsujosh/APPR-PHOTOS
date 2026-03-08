"""Device selection and hardware capability detection.

Priority when device='auto':
  1. CUDA  — NVIDIA GPU (supports AMP, multi-GPU)
  2. MPS   — Apple Silicon GPU (M1/M2/M3/M4)
  3. CPU   — fallback

Explicit values accepted: 'cuda', 'cuda:0', 'cuda:1', 'mps', 'cpu'
"""

import logging
import torch

logger = logging.getLogger("aapr")


# ------------------------------------------------------------------ #
# Device selection                                                    #
# ------------------------------------------------------------------ #

def get_device(preferred: str = "auto") -> torch.device:
    """Return the best available device.

    Args:
        preferred: 'auto', 'cuda', 'cuda:N', 'mps', or 'cpu'
    """
    if preferred == "auto":
        device = _auto_select()
    else:
        device = torch.device(preferred)
        _validate(device)

    _log_device_info(device)
    return device


def _auto_select() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def _validate(device: torch.device):
    """Raise informative error if requested device is not usable."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{device}' requested but CUDA is not available. "
                "Check your PyTorch installation and driver version."
            )
        n = torch.cuda.device_count()
        idx = device.index or 0
        if idx >= n:
            raise RuntimeError(
                f"CUDA device index {idx} requested but only {n} GPU(s) found."
            )
    elif device.type == "mps":
        if not _mps_available():
            raise RuntimeError(
                "MPS device requested but not available. "
                "Requires Apple Silicon (M1+) and PyTorch >= 1.12 built with MPS."
            )


# ------------------------------------------------------------------ #
# Device info                                                         #
# ------------------------------------------------------------------ #

def _log_device_info(device: torch.device):
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        mem_gb = props.total_memory / 1024 ** 3
        logger.info(
            f"Device: cuda:{idx} — {props.name} "
            f"({mem_gb:.1f} GB, {props.multi_processor_count} SMs, "
            f"CUDA {torch.version.cuda})"
        )
        if torch.cuda.device_count() > 1:
            logger.info(f"  {torch.cuda.device_count()} GPUs available; using cuda:{idx}")
    elif device.type == "mps":
        import platform
        logger.info(f"Device: mps — Apple Silicon ({platform.processor()})")
    else:
        import platform
        logger.info(f"Device: cpu — {platform.processor() or platform.machine()}")


# ------------------------------------------------------------------ #
# AMP / mixed precision helpers                                       #
# ------------------------------------------------------------------ #

def get_autocast_context(device: torch.device):
    """Return a torch.autocast context appropriate for the device.

    - CUDA: float16 AMP (significant speedup on Ampere+)
    - MPS:  bfloat16 AMP (available on PyTorch >= 2.4 with MPS)
    - CPU:  bfloat16 AMP (useful on AVX-512 machines, no-op otherwise)

    Usage:
        with get_autocast_context(device):
            output = model(input)
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    elif device.type == "mps":
        # bfloat16 on MPS requires torch >= 2.4; fall back silently if not supported
        try:
            return torch.autocast(device_type="mps", dtype=torch.bfloat16)
        except Exception:
            return _NoOpContext()
    else:
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def get_grad_scaler(device: torch.device) -> "torch.cuda.amp.GradScaler | None":
    """Return a GradScaler for CUDA AMP; None for MPS/CPU (not needed)."""
    if device.type == "cuda":
        return torch.amp.GradScaler()
    return None


def supports_amp(device: torch.device) -> bool:
    """Whether the device meaningfully benefits from AMP."""
    return device.type in ("cuda", "mps")


class _NoOpContext:
    """Context manager that does nothing — fallback when AMP is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ------------------------------------------------------------------ #
# Safe checkpoint loading                                             #
# ------------------------------------------------------------------ #

def load_checkpoint(path, device: torch.device, **kwargs) -> dict:
    """Load a checkpoint with correct map_location for any device.

    MPS requires loading to CPU first then moving tensors, because
    torch.load with map_location='mps' can fail on older PyTorch builds.
    """
    if device.type == "mps":
        state = torch.load(path, map_location="cpu", **kwargs)
    else:
        state = torch.load(path, map_location=device, **kwargs)
    return state
