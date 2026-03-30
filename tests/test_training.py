"""Test training loop on synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel
from aapr.models.adversary import MultiHeadAdversary
from aapr.training.losses import CombinedLoss, AdversaryLoss
from aapr.training.schedulers import LambdaScheduler, AdversaryRefreshScheduler
from aapr.training.metrics import MetricTracker, compute_uar


def make_synthetic_loader(n=64, feat_dim=128, seq_len=32, batch_size=16):
    """Create a synthetic dataloader that mimics real data format."""
    features = torch.randn(n, feat_dim, seq_len)
    utility_labels = torch.randint(0, 6, (n,))
    speaker_ids = torch.randint(0, 10, (n,))
    genders = torch.randint(0, 2, (n,))

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return n
        def __getitem__(self, idx):
            return {
                "features": features[idx],
                "utility_label": utility_labels[idx],
                "privacy_labels": {
                    "speaker_id": speaker_ids[idx],
                    "gender": genders[idx],
                },
            }

    def collate(batch):
        return {
            "features": torch.stack([b["features"] for b in batch]),
            "utility_label": torch.stack([b["utility_label"] for b in batch]),
            "privacy_labels": {
                "speaker_id": torch.stack([b["privacy_labels"]["speaker_id"] for b in batch]),
                "gender": torch.stack([b["privacy_labels"]["gender"] for b in batch]),
            },
        }

    return DataLoader(SyntheticDataset(), batch_size=batch_size, collate_fn=collate, drop_last=True)


class TestLosses:
    def test_combined_loss(self):
        criterion = CombinedLoss(lambda_privacy=1.0)
        utility_logits = torch.randn(8, 6, requires_grad=True)
        utility_labels = torch.randint(0, 6, (8,))
        privacy_logits = {"gender": torch.randn(8, 2, requires_grad=True), "speaker_id": torch.randn(8, 10, requires_grad=True)}
        privacy_labels = {"gender": torch.randint(0, 2, (8,)), "speaker_id": torch.randint(0, 10, (8,))}
        kl = torch.tensor(0.01)

        losses = criterion(utility_logits, utility_labels, privacy_logits, privacy_labels, kl)
        assert "total" in losses
        assert "utility" in losses
        assert "privacy" in losses
        assert losses["total"].requires_grad

    def test_invalid_labels_skipped(self):
        criterion = CombinedLoss()
        privacy_logits = {"gender": torch.randn(4, 2, requires_grad=True)}
        privacy_labels = {"gender": torch.tensor([-1, -1, -1, -1])}
        losses = criterion(
            torch.randn(4, 6, requires_grad=True), torch.randint(0, 6, (4,)),
            privacy_logits, privacy_labels, torch.tensor(0.0),
        )
        # When all privacy labels are invalid, total still computes from utility
        assert losses["total"].requires_grad


class TestSchedulers:
    def test_lambda_warmup(self):
        sched = LambdaScheduler(target_lambda=2.0, total_epochs=10)
        assert sched.get_lambda(0) == pytest.approx(0.0)
        assert 0.0 < sched.get_lambda(5) < 2.0
        assert sched.get_lambda(10) == pytest.approx(2.0, rel=1e-3)
        assert sched.get_lambda(20) == pytest.approx(2.0, rel=1e-3)

    def test_adversary_refresh(self):
        sched = AdversaryRefreshScheduler(refresh_interval=20, retrain_epochs=5)
        assert not sched.should_refresh(0)
        assert sched.should_refresh(20)
        assert sched.should_refresh(40)
        assert not sched.should_refresh(10)


class TestMetrics:
    def test_metric_tracker(self):
        tracker = MetricTracker()
        tracker.update(
            torch.randn(8, 6), torch.randint(0, 6, (8,)),
            {"gender": torch.randn(8, 2)}, {"gender": torch.randint(0, 2, (8,))},
            loss=1.0,
        )
        metrics = tracker.compute()
        assert "utility_uar" in metrics
        assert "utility_wa" in metrics
        assert 0 <= metrics["utility_uar"] <= 1

    def test_uar_perfect(self):
        import numpy as np
        y = np.array([0, 1, 2, 0, 1, 2])
        assert compute_uar(y, y) == 1.0


class TestTrainerOneEpoch:
    def test_one_epoch_runs(self, tmp_path):
        from aapr.training.trainer import Trainer

        loader = make_synthetic_loader()
        pf = PrivacyFilter(input_dim=128, output_dim=64, num_layers=2)
        tm = TaskModel(input_dim=64, num_classes=6)
        adv = MultiHeadAdversary(input_dim=64, heads={"gender": 2, "speaker_id": 10})

        trainer = Trainer(
            pf, tm, adv,
            device=torch.device("cpu"),
            lambda_privacy=1.0,
            lambda_warmup_epochs=2,
            adversary_refresh_interval=0,
            checkpoint_dir=str(tmp_path),
            use_cached_features=True,
        )

        metrics = trainer.train_epoch(loader, epoch=0)
        assert "utility_uar" in metrics
        assert "loss" in metrics

    def test_fit_two_epochs(self, tmp_path):
        from aapr.training.trainer import Trainer

        loader = make_synthetic_loader()
        pf = PrivacyFilter(input_dim=128, output_dim=64, num_layers=2)
        tm = TaskModel(input_dim=64, num_classes=6)
        adv = MultiHeadAdversary(input_dim=64, heads={"gender": 2, "speaker_id": 10})

        trainer = Trainer(
            pf, tm, adv,
            device=torch.device("cpu"),
            checkpoint_dir=str(tmp_path),
            use_cached_features=True,
        )

        final = trainer.fit(loader, loader, num_epochs=2)
        assert "utility_uar" in final
        # Check checkpoint saved
        assert (tmp_path / "best_model.pt").exists()
