import random

import torch
from torch_geometric.data import Data

from clouds.transforms.color import RandomColorAutoContrast


class TestRandomColorAutoContrast:
    def test_transform_applies_with_probability(self):
        """Test that transform is applied roughly p% of the time."""
        # Use a high p to ensure it applies
        transform = RandomColorAutoContrast(p=1.0)

        # Create test data with known colors
        color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8], [0.1, 0.5, 0.9]])
        data = Data(color=color, pos=torch.randn(3, 3))

        transformed = transform(data)

        # Should be different from original
        assert not torch.allclose(transformed.color, color)

    def test_transform_does_not_apply_when_p_is_zero(self):
        """Test that transform doesn't apply when p=0."""
        transform = RandomColorAutoContrast(p=0.0)

        color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        data = Data(color=color, pos=torch.randn(2, 3))

        transformed = transform(data)

        # Should be identical
        assert torch.allclose(transformed.color, color)

    def test_auto_contrast_behavior(self):
        """Test that the transform correctly normalizes colors."""
        # Fix random seed for reproducibility
        torch.manual_seed(42)
        random.seed(42)

        transform = RandomColorAutoContrast(p=1.0, blend_factor=1.0)

        # Create data with colors in a specific range
        color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8], [0.8, 0.9, 0.5]])
        data = Data(color=color, pos=torch.randn(3, 3))

        transformed = transform(data)

        # With blend_factor=1.0, colors should be normalized to [0,1] range
        # Check that min is close to 0 and max is close to 1 for each channel
        colmin = transformed.color.min(dim=0)[0]
        colmax = transformed.color.max(dim=0)[0]

        # Allow small epsilon tolerance
        assert torch.allclose(colmin, torch.zeros(3), atol=1e-6)
        assert torch.allclose(colmax, torch.ones(3), atol=1e-6)

    def test_blend_factor_effect(self):
        """Test that blend_factor controls mixing between original and normalized."""
        transform = RandomColorAutoContrast(p=1.0, blend_factor=0.5)

        color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        data = Data(color=color, pos=torch.randn(2, 3))

        transformed = transform(data)

        # With blend_factor=0.5, result should be exactly halfway between original and normalized
        colmin = color.min(dim=0, keepdim=True)[0]
        colmax = color.max(dim=0, keepdim=True)[0]
        scale = 1 / (1e-7 + colmax - colmin)
        normalized = scale * color - colmin * scale
        expected = 0.5 * color + 0.5 * normalized

        assert torch.allclose(transformed.color, expected)

    def test_random_blend_factor(self):
        """Test that when blend_factor is None, a random value is used."""
        transform = RandomColorAutoContrast(p=1.0, blend_factor=None)

        color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        data1 = Data(color=color.clone(), pos=torch.randn(2, 3))
        data2 = Data(color=color.clone(), pos=torch.randn(2, 3))

        # Fix seed for first call
        torch.manual_seed(42)
        random.seed(42)
        transformed1 = transform(data1)

        # Reset seed for second call - should get same result
        torch.manual_seed(42)
        random.seed(42)
        transformed2 = transform(data2)

        assert torch.allclose(transformed1.color, transformed2.color)

    def test_multiple_channels(self):
        """Test that transform works correctly with different number of channels."""
        transform = RandomColorAutoContrast(p=1.0, blend_factor=1.0)

        # RGB (3 channels)
        color_rgb = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
        data_rgb = Data(color=color_rgb, pos=torch.randn(2, 3))
        transformed_rgb = transform(data_rgb)
        assert transformed_rgb.color.shape == (2, 3)

        # Single channel (grayscale)
        color_gray = torch.tensor([[0.2], [0.6]])
        data_gray = Data(color=color_gray, pos=torch.randn(2, 3))
        transformed_gray = transform(data_gray)
        assert transformed_gray.color.shape == (2, 1)
