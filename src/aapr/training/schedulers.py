class LambdaScheduler:
    """Warmup schedule for privacy penalty lambda.

    Linearly ramps from 0 to target_lambda over warmup_epochs.
    """

    def __init__(self, target_lambda: float = 1.0, warmup_epochs: int = 10):
        self.target_lambda = target_lambda
        self.warmup_epochs = warmup_epochs

    def get_lambda(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.target_lambda
        progress = min(epoch / self.warmup_epochs, 1.0)
        return self.target_lambda * progress


class AdversaryRefreshScheduler:
    """Manages adversary refresh cycles.

    Every refresh_interval epochs:
    1. Reset adversary parameters
    2. Run adversary-only training for retrain_epochs
    3. Resume joint training
    """

    def __init__(self, refresh_interval: int = 20, retrain_epochs: int = 5):
        self.refresh_interval = refresh_interval
        self.retrain_epochs = retrain_epochs

    def should_refresh(self, epoch: int) -> bool:
        if self.refresh_interval <= 0:
            return False
        return epoch > 0 and epoch % self.refresh_interval == 0

    def is_retrain_phase(self, epoch: int) -> bool:
        """Check if current epoch is in adversary-only retrain phase."""
        if self.refresh_interval <= 0:
            return False
        cycle_pos = epoch % self.refresh_interval
        last_refresh = epoch - cycle_pos
        if last_refresh == 0:
            return False
        return cycle_pos < self.retrain_epochs
