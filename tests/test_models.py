"""Test model architectures: forward pass shapes, gradient flow."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from aapr.models.gradient_reversal import GradientReversalLayer
from aapr.models.privacy_filter import PrivacyFilter
from aapr.models.task_model import TaskModel, AttentionPooling
from aapr.models.adversary import MultiHeadAdversary


@pytest.fixture
def batch_size():
    return 4

@pytest.fixture
def seq_len():
    return 32

@pytest.fixture
def input_dim():
    return 128


class TestGRL:
    def test_forward_identity(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 10)
        out = grl(x)
        assert torch.allclose(out, x)

    def test_gradient_reversal(self):
        grl = GradientReversalLayer(lambda_=1.0)
        x = torch.randn(4, 10, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()
        # Gradient should be negated
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_gradient_scaling(self):
        grl = GradientReversalLayer(lambda_=2.0)
        x = torch.randn(4, 10, requires_grad=True)
        out = grl(x)
        loss = out.sum()
        loss.backward()
        assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))


class TestPrivacyFilter:
    def test_output_shape(self, batch_size, seq_len, input_dim):
        pf = PrivacyFilter(input_dim=input_dim, hidden_dim=256, output_dim=64)
        x = torch.randn(batch_size, input_dim, seq_len)
        z, kl = pf(x)
        assert z.shape == (batch_size, 64, seq_len)
        assert kl.shape == ()

    def test_no_vib(self, batch_size, seq_len, input_dim):
        pf = PrivacyFilter(input_dim=input_dim, use_vib=False)
        x = torch.randn(batch_size, input_dim, seq_len)
        z, kl = pf(x)
        assert kl.item() == 0.0

    def test_vib_kl_positive(self, batch_size, seq_len, input_dim):
        pf = PrivacyFilter(input_dim=input_dim, use_vib=True)
        pf.train()
        x = torch.randn(batch_size, input_dim, seq_len)
        _, kl = pf(x)
        assert kl.item() >= 0.0


class TestAttentionPooling:
    def test_output_shape(self, batch_size, seq_len, input_dim):
        pool = AttentionPooling(input_dim)
        x = torch.randn(batch_size, input_dim, seq_len)
        out = pool(x)
        assert out.shape == (batch_size, input_dim)


class TestTaskModel:
    def test_output_shape(self, batch_size, seq_len):
        tm = TaskModel(input_dim=128, num_classes=6)
        x = torch.randn(batch_size, 128, seq_len)
        logits = tm(x)
        assert logits.shape == (batch_size, 6)


class TestAdversary:
    def test_output_shape(self, batch_size, seq_len):
        adv = MultiHeadAdversary(
            input_dim=128,
            heads={"gender": 2, "speaker_id": 10},
        )
        x = torch.randn(batch_size, 128, seq_len)
        out = adv(x)
        assert out["gender"].shape == (batch_size, 2)
        assert out["speaker_id"].shape == (batch_size, 10)

    def test_reset_parameters(self):
        adv = MultiHeadAdversary(input_dim=64, heads={"gender": 2})
        w_before = adv.heads["gender"].weight.clone()
        adv.reset_parameters()
        # Weights should change after reset
        assert not torch.allclose(w_before, adv.heads["gender"].weight)

    def test_gradient_flow_through_grl(self, batch_size, seq_len):
        """Verify gradients flow reversed through GRL to upstream params."""
        pf = PrivacyFilter(input_dim=64, output_dim=64, use_vib=False)
        adv = MultiHeadAdversary(input_dim=64, heads={"gender": 2})

        x = torch.randn(batch_size, 64, seq_len, requires_grad=True)
        z, _ = pf(x)
        logits = adv(z)

        loss = logits["gender"].sum()
        loss.backward()

        # Filter params should have gradients (reversed by GRL)
        for p in pf.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestFullPipeline:
    def test_end_to_end(self, batch_size, seq_len, input_dim):
        from aapr.features.image_cnn import ImageCNNExtractor
        from aapr.models.full_system import FullSystem

        fe = ImageCNNExtractor(output_dim=input_dim)
        pf = PrivacyFilter(input_dim=input_dim, output_dim=64)
        tm = TaskModel(input_dim=64, num_classes=6)
        adv = MultiHeadAdversary(input_dim=64, heads={"gender": 2, "speaker_id": 10})

        system = FullSystem(fe, pf, tm, adv)

        image = torch.randn(batch_size, 3, 224, 224)
        out = system(image)

        assert out["utility_logits"].shape == (batch_size, 6)
        assert out["privacy_logits"]["gender"].shape == (batch_size, 2)
        assert out["privacy_logits"]["speaker_id"].shape == (batch_size, 10)
        assert out["kl_loss"].shape == ()
