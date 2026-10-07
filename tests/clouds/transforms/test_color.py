import random

import torch
from torch_geometric.data import Data

from clouds.transforms.color import RandomColorAutoContrast


def test_transform_applies_with_probability():
    color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8], [0.1, 0.5, 0.9]])
    data = Data(color=color, pos=torch.randn(3, 3))

    assert not torch.allclose(RandomColorAutoContrast(p=1.0)(data).color, color)


def test_transform_does_not_apply_when_p_is_zero():
    color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    data = Data(color=color, pos=torch.randn(2, 3))

    assert torch.allclose(RandomColorAutoContrast(p=0.0)(data).color, color)


def test_auto_contrast_behavior():
    color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8], [0.8, 0.9, 0.5]])
    transformed = RandomColorAutoContrast(p=1.0, blend_factor=1.0)(Data(color=color, pos=torch.randn(3, 3)))

    # blend_factor=1.0 normalizes each channel to [0, 1].
    assert torch.allclose(transformed.color.min(dim=0)[0], torch.zeros(3), atol=1e-6)
    assert torch.allclose(transformed.color.max(dim=0)[0], torch.ones(3), atol=1e-6)


def test_blend_factor_effect():
    color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])
    colmin = color.min(dim=0, keepdim=True)[0]
    scale = 1 / (1e-7 + color.max(dim=0, keepdim=True)[0] - colmin)
    expected = 0.5 * color + 0.5 * (scale * color - colmin * scale)

    assert torch.allclose(
        RandomColorAutoContrast(p=1.0, blend_factor=0.5)(Data(color=color, pos=torch.randn(2, 3))).color, expected
    )


def test_random_blend_factor():
    color = torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]])

    torch.manual_seed(42)
    random.seed(42)
    transformed1 = RandomColorAutoContrast(p=1.0, blend_factor=None)(Data(color=color.clone(), pos=torch.randn(2, 3)))

    torch.manual_seed(42)
    random.seed(42)
    transformed2 = RandomColorAutoContrast(p=1.0, blend_factor=None)(Data(color=color.clone(), pos=torch.randn(2, 3)))

    assert torch.allclose(transformed1.color, transformed2.color)


def test_multiple_channels():
    transform = RandomColorAutoContrast(p=1.0, blend_factor=1.0)

    assert transform(Data(color=torch.tensor([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]]), pos=torch.randn(2, 3))).color.shape == (2, 3)
    assert transform(Data(color=torch.tensor([[0.2], [0.6]]), pos=torch.randn(2, 3))).color.shape == (2, 1)
