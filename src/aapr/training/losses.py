import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Knowledge distillation loss: soft targets from teacher + hard CE from labels.

    Total = alpha * KL(student || teacher) * T^2  +  (1 - alpha) * CE(student, labels)
    """

    def __init__(self, alpha: float = 0.7, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        T = self.temperature
        soft_student = F.log_softmax(student_logits / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T ** 2)
        ce_loss = F.cross_entropy(student_logits, labels)
        return self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss


class CombinedLoss(nn.Module):
    """Combined utility + VIB KL - lambda*privacy loss.

    If teacher logits are provided, utility loss uses knowledge distillation.
    GRL inside the adversary handles sign reversal for filter training.
    """

    def __init__(
        self,
        lambda_privacy: float = 1.0,
        distillation_alpha: float = 0.0,
        distillation_temperature: float = 4.0,
    ):
        super().__init__()
        self.lambda_privacy = lambda_privacy
        self.utility_criterion = nn.CrossEntropyLoss()
        self.privacy_criterion = nn.CrossEntropyLoss()
        self.distillation = DistillationLoss(distillation_alpha, distillation_temperature)
        self.use_distillation = distillation_alpha > 0.0

    def set_lambda(self, lambda_: float):
        self.lambda_privacy = lambda_

    def forward(
        self,
        utility_logits: torch.Tensor,
        utility_labels: torch.Tensor,
        privacy_logits: dict[str, torch.Tensor],
        privacy_labels: dict[str, torch.Tensor],
        kl_loss: torch.Tensor,
        teacher_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Utility loss (distillation if teacher available, else plain CE)
        if self.use_distillation and teacher_logits is not None:
            utility_loss = self.distillation(utility_logits, teacher_logits, utility_labels)
        else:
            utility_loss = self.utility_criterion(utility_logits, utility_labels)

        # Privacy loss (average over all valid heads)
        privacy_loss = torch.tensor(0.0, device=utility_logits.device)
        privacy_losses = {}
        n_heads = 0
        for attr_name, logits in privacy_logits.items():
            labels = privacy_labels.get(attr_name)
            if labels is None:
                continue
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            head_loss = self.privacy_criterion(logits[valid_mask], labels[valid_mask])
            privacy_losses[attr_name] = head_loss
            privacy_loss = privacy_loss + head_loss
            n_heads += 1

        if n_heads > 0:
            privacy_loss = privacy_loss / n_heads

        # Total: utility + KL + privacy (GRL in adversary handles sign for filter)
        total = utility_loss + kl_loss + self.lambda_privacy * privacy_loss

        return {
            "total": total,
            "utility": utility_loss,
            "privacy": privacy_loss,
            "kl": kl_loss,
            **{f"privacy_{k}": v for k, v in privacy_losses.items()},
        }


class AdversaryLoss(nn.Module):
    """Standalone adversary loss for adversary-only retraining phase."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        privacy_logits: dict[str, torch.Tensor],
        privacy_labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=next(iter(privacy_logits.values())).device)
        n = 0
        for attr_name, logits in privacy_logits.items():
            labels = privacy_labels.get(attr_name)
            if labels is None:
                continue
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            total = total + self.criterion(logits[valid_mask], labels[valid_mask])
            n += 1
        return total / max(n, 1)
