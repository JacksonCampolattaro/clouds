"""Smoke tests for the core neural-network building blocks.

These only check shape bookkeeping and the wiring semantics -- the errors that
tend to break every model at once -- using tiny CPU tensors.
"""

import pytest
import torch
from torch import nn

from clouds.nn.activation import FINER
from clouds.nn.dropout import PathDropout, PointDropout
from clouds.nn.functional import CombinedLoss
from clouds.nn.models import MLP
from clouds.nn.parallel import Parallel
from clouds.nn.residual import Residual
from clouds.nn.resolver import activation_resolver
from clouds.nn.sequential import Sequential


class Scale(nn.Module):
    """A trivial channel-preserving module with the usual channel attributes."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor
        self.in_channels = 3
        self.out_channels = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


def test_mlp_infers_channel_counts_and_output_shape():
    x = torch.randn(7, 3)
    mlp = MLP(in_channels=3, hidden_channels=8, out_channels=4, num_layers=3)
    assert mlp(x).shape == (7, 4)
    assert mlp.in_channels == 3
    assert mlp.out_channels == 4


def test_mlp_explicit_channel_list_output_shape():
    mlp = MLP(channel_list=[3, 5, 2], norm=None, act='relu')
    assert mlp(torch.randn(4, 3)).shape == (4, 2)


def test_sequential_chains_modules():
    seq = Sequential(Scale(2.0), Scale(3.0))
    assert torch.equal(seq(torch.ones(2, 3)), torch.full((2, 3), 6.0))


def test_parallel_sums_branches():
    par = Parallel(Scale(2.0), Scale(3.0))
    assert torch.equal(par(torch.ones(2, 3)), torch.full((2, 3), 5.0))


def test_residual_with_identity_path_is_sum():
    res = Residual(Scale(1.0))
    out = res(torch.ones(2, 3))
    assert out.shape == (2, 3)
    assert torch.equal(out, torch.full((2, 3), 2.0))


def test_combined_loss_applies_weights():
    loss = CombinedLoss([nn.MSELoss(), nn.L1Loss()], weights=[2.0, 1.0])
    assert loss(torch.zeros(4), torch.ones(4)).item() == pytest.approx(3.0)


def test_point_dropout_is_identity_in_eval():
    x = torch.randn(5, 3)
    assert torch.equal(PointDropout(p=0.5).eval()(x), x)


def test_path_dropout_is_identity_in_eval():
    x = torch.randn(5, 3)
    batch = torch.zeros(5, dtype=torch.long)
    assert torch.equal(PathDropout(p=0.5).eval()(x, batch), x)


def test_activation_resolver_resolves_strings_and_custom_layers():
    assert isinstance(activation_resolver('relu'), nn.ReLU)
    assert isinstance(activation_resolver('finer'), FINER)
