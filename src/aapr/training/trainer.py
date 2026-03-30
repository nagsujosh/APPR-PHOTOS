import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CombinedLoss, AdversaryLoss
from .metrics import MetricTracker
from .schedulers import LambdaScheduler, AdversaryRefreshScheduler
from ..utils.device import get_autocast_context, get_grad_scaler, load_checkpoint

logger = logging.getLogger("aapr")


class Trainer:
    """Adversarial min-max training with optional student-teacher distillation.

    Architecture:
      - feature_extractor: image encoder (may be frozen)
      - teacher: frozen model trained on raw features, provides soft targets
      - privacy_filter + task_model: student, trained adversarially
      - adversary: multi-head classifier with GRL; reset every refresh_interval epochs
    """

    def __init__(
        self,
        privacy_filter: nn.Module,
        task_model: nn.Module,
        adversary: nn.Module,
        feature_extractor: nn.Module | None = None,
        teacher: nn.Module | None = None,
        device: torch.device = torch.device("cpu"),
        lr_main: float = 1e-3,
        lr_adversary: float = 5e-4,
        lambda_privacy: float = 1.0,
        lambda_warmup_epochs: int = 10,
        adversary_refresh_interval: int = 20,
        adversary_retrain_epochs: int = 5,
        checkpoint_dir: str = "outputs/checkpoints",
        use_cached_features: bool = False,
        distillation_alpha: float = 0.0,
        distillation_temperature: float = 4.0,
        grad_clip: float = 1.0,
        num_epochs: int = 100,
    ):
        self.privacy_filter = privacy_filter.to(device)
        self.task_model = task_model.to(device)
        self.adversary = adversary.to(device)
        self.feature_extractor = feature_extractor.to(device) if feature_extractor else None
        self.teacher = teacher.to(device) if teacher else None
        self.device = device
        self.use_cached_features = use_cached_features
        self.grad_clip = grad_clip

        # AMP: GradScaler for CUDA (float16), None for MPS/CPU
        self.scaler = get_grad_scaler(device)

        # Optimizers
        main_params = list(privacy_filter.parameters()) + list(task_model.parameters())
        if feature_extractor:
            main_params += [p for p in feature_extractor.parameters() if p.requires_grad]
        self.optimizer_main = torch.optim.Adam(main_params, lr=lr_main)
        self.optimizer_adv = torch.optim.Adam(adversary.parameters(), lr=lr_adversary)

        # Cosine LR decay for main optimizer over full training
        self.scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_main, T_max=num_epochs, eta_min=lr_main * 0.01
        )

        # Loss
        self.criterion = CombinedLoss(
            lambda_privacy, distillation_alpha, distillation_temperature
        )
        self.adv_criterion = AdversaryLoss()

        # Schedulers
        self.lambda_scheduler = LambdaScheduler(lambda_privacy, num_epochs)
        self.refresh_scheduler = AdversaryRefreshScheduler(
            adversary_refresh_interval, adversary_retrain_epochs
        )

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_uar = 0.0

    def _extract_features(self, batch):
        if self.use_cached_features:
            return batch["features"].to(self.device)
        image = batch["image"].to(self.device)
        if self.feature_extractor:
            return self.feature_extractor(image)
        return image

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.privacy_filter.train()
        self.task_model.train()
        self.adversary.train()

        # Update lambda
        current_lambda = self.lambda_scheduler.get_lambda(epoch)
        self.criterion.set_lambda(current_lambda)
        self.adversary.set_lambda(current_lambda)

        # Check adversary refresh
        if self.refresh_scheduler.should_refresh(epoch):
            logger.info(f"Epoch {epoch}: Refreshing adversary parameters")
            self.adversary.reset_parameters()
            self.adversary = self.adversary.to(self.device)
            self.optimizer_adv = torch.optim.Adam(
                self.adversary.parameters(), lr=self.optimizer_adv.defaults["lr"]
            )

        is_retrain = self.refresh_scheduler.is_retrain_phase(epoch)
        tracker = MetricTracker()

        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            features = self._extract_features(batch)
            utility_labels = batch["utility_label"].to(self.device)
            privacy_labels = {k: v.to(self.device) for k, v in batch["privacy_labels"].items()}

            # Teacher soft targets (no grad — teacher is frozen)
            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad():
                    with get_autocast_context(self.device):
                        teacher_logits = self.teacher(features)

            # Forward pass through student (AMP where available)
            with get_autocast_context(self.device):
                filtered, kl_loss = self.privacy_filter(features)
                utility_logits = self.task_model(filtered)
                privacy_logits = self.adversary(filtered)

            if is_retrain:
                # Adversary-only phase: zero both to avoid stale filter gradients
                with get_autocast_context(self.device):
                    adv_loss = self.adv_criterion(privacy_logits, privacy_labels)
                self.optimizer_main.zero_grad()
                self.optimizer_adv.zero_grad()
                if self.scaler:
                    self.scaler.scale(adv_loss).backward()
                    self.scaler.step(self.optimizer_adv)
                    self.scaler.update()
                else:
                    adv_loss.backward()
                    self.optimizer_adv.step()
                loss_val = adv_loss.item()
            else:
                # Joint training: filter+task minimize utility+KL, maximize privacy (via GRL)
                with get_autocast_context(self.device):
                    losses = self.criterion(
                        utility_logits, utility_labels,
                        privacy_logits, privacy_labels, kl_loss,
                        teacher_logits=teacher_logits,
                    )
                self.optimizer_main.zero_grad()
                self.optimizer_adv.zero_grad()
                if self.scaler:
                    self.scaler.scale(losses["total"]).backward()
                    self.scaler.unscale_(self.optimizer_main)
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            list(self.privacy_filter.parameters()) +
                            list(self.task_model.parameters()),
                            self.grad_clip,
                        )
                    self.scaler.step(self.optimizer_main)
                    self.scaler.step(self.optimizer_adv)
                    self.scaler.update()
                else:
                    losses["total"].backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            list(self.privacy_filter.parameters()) +
                            list(self.task_model.parameters()),
                            self.grad_clip,
                        )
                    self.optimizer_main.step()
                    self.optimizer_adv.step()
                loss_val = losses["total"].item()

            tracker.update(
                utility_logits.detach(), utility_labels,
                privacy_logits, privacy_labels, loss_val,
            )

        # Step LR scheduler once per epoch (only during joint training epochs)
        if not is_retrain:
            self.scheduler_main.step()

        metrics = tracker.compute()
        metrics["lambda"] = current_lambda
        metrics["is_retrain"] = is_retrain
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.privacy_filter.eval()
        self.task_model.eval()
        self.adversary.eval()

        tracker = MetricTracker()

        for batch in loader:
            features = self._extract_features(batch)
            utility_labels = batch["utility_label"].to(self.device)
            privacy_labels = {k: v.to(self.device) for k, v in batch["privacy_labels"].items()}

            with get_autocast_context(self.device):
                filtered, kl_loss = self.privacy_filter(features)
                utility_logits = self.task_model(filtered)
                privacy_logits = self.adversary(filtered)

                losses = self.criterion(
                    utility_logits, utility_labels,
                    privacy_logits, privacy_labels, kl_loss,
                )

            tracker.update(
                utility_logits, utility_labels,
                privacy_logits, privacy_labels, losses["total"].item(),
            )

        return tracker.compute()

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "privacy_filter": self.privacy_filter.state_dict(),
            "task_model": self.task_model.state_dict(),
            "adversary": self.adversary.state_dict(),
            "optimizer_main": self.optimizer_main.state_dict(),
            "optimizer_adv": self.optimizer_adv.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, self.checkpoint_dir / f"checkpoint_epoch{epoch}.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")

    def load_checkpoint(self, path: str | Path):
        state = load_checkpoint(path, self.device, weights_only=False)
        self.privacy_filter.load_state_dict(state["privacy_filter"])
        self.task_model.load_state_dict(state["task_model"])
        self.adversary.load_state_dict(state["adversary"])
        self.optimizer_main.load_state_dict(state["optimizer_main"])
        self.optimizer_adv.load_state_dict(state["optimizer_adv"])
        return state["epoch"], state.get("metrics", {})

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        writer=None,
    ) -> dict:
        val_metrics = {}
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            retrain_marker = " [ADV RETRAIN]" if train_metrics.get("is_retrain") else ""
            logger.info(
                f"Epoch {epoch}{retrain_marker} | "
                f"Train UAR: {train_metrics['utility_uar']:.4f} | "
                f"Val UAR: {val_metrics['utility_uar']:.4f} | "
                f"Lambda: {train_metrics['lambda']:.3f}"
            )

            if writer:
                for k, v in train_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"val/{k}", v, epoch)

            is_best = val_metrics["utility_uar"] > self.best_val_uar
            if is_best:
                self.best_val_uar = val_metrics["utility_uar"]
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

        return val_metrics
