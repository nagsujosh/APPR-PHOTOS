import math


class LambdaScheduler:
    """DANN-style sigmoid schedule for privacy penalty lambda.

    λ(p) = target · (2 / (1 + exp(-10·p)) - 1)   where p = epoch / total_epochs

    This gives a slow start (natural warmup), fast growth through the middle,
    and a smooth plateau — empirically more stable than a linear ramp.
    Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016).
    """

    def __init__(self, target_lambda: float = 1.0, total_epochs: int = 100):
        self.target_lambda = target_lambda
        self.total_epochs  = max(total_epochs, 1)

    def get_lambda(self, epoch: int) -> float:
        p = epoch / self.total_epochs
        return self.target_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


class AdversaryRefreshScheduler:
    """Manages adversary refresh cycles.

    Every refresh_interval epochs:
    1. Reset adversary parameters
    2. Run adversary-only training for retrain_epochs
    3. Resume joint training
    """

    def __init__(self, refresh_interval: int = 20, retrain_epochs: int = 5):
        self.refresh_interval = refresh_interval
        self.retrain_epochs   = retrain_epochs

    def should_refresh(self, epoch: int) -> bool:
        if self.refresh_interval <= 0:
            return False
        return epoch > 0 and epoch % self.refresh_interval == 0

    def is_retrain_phase(self, epoch: int) -> bool:
        """Check if current epoch is in adversary-only retrain phase."""
        if self.refresh_interval <= 0:
            return False
        cycle_pos   = epoch % self.refresh_interval
        last_refresh = epoch - cycle_pos
        if last_refresh == 0:
            return False
        return cycle_pos < self.retrain_epochs
