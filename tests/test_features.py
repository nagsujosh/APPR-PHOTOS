"""Test feature extractors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from aapr.features.image_cnn import ImageCNNExtractor


class TestImageCNNExtractor:
    def test_output_shape(self):
        extractor = ImageCNNExtractor(output_dim=128)
        image = torch.randn(2, 3, 224, 224)
        features = extractor(image)
        assert features.shape[0] == 2
        assert features.shape[1] == 128
        assert features.shape[2] > 0

    def test_output_dim(self):
        extractor = ImageCNNExtractor(output_dim=80)
        assert extractor.output_dim == 80

    def test_finite_values(self):
        extractor = ImageCNNExtractor()
        image = torch.rand(1, 3, 224, 224)
        features = extractor(image)
        assert torch.isfinite(features).all()
