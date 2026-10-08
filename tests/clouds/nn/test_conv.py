"""Smoke tests for the graph convolution layers on tiny synthetic graphs.

These check channel bookkeeping and both edge conventions used in this
codebase: PyG-style pairwise ``(2, E)`` indices and source-indexed ``(N, K)``
indices. No datasets are required.
"""

import pytest
import torch

from clouds.nn.conv.aggr import (
    AggrConv,
    AggrLinConv,
    LinAggrConv,
    MLPAggrConv,
    ProjAggrConv,
    ProjAggrProjConv,
)
from clouds.nn.conv.pointnet import SimplePointNetConv
from clouds.nn.conv.selection import ProjSelectionConv, SelectionConv

NUM_NODES = 6
IN_CHANNELS = 4
OUT_CHANNELS = 5


@pytest.fixture
def graph():
    x = torch.randn(NUM_NODES, IN_CHANNELS)
    pos = torch.randn(NUM_NODES, 3)
    edge_index = torch.stack([torch.arange(NUM_NODES - 1), torch.arange(1, NUM_NODES)])
    return x, pos, edge_index


def _source_indexed(num_nodes: int = NUM_NODES, k: int = 2) -> torch.Tensor:
    return torch.randint(0, num_nodes, (num_nodes, k))


def test_aggr_conv_handles_both_edge_conventions(graph):
    x, _, edge_index = graph
    conv = AggrConv(IN_CHANNELS)
    assert conv(x, edge_index).shape == (NUM_NODES, IN_CHANNELS)
    assert conv(x, _source_indexed()).shape == (NUM_NODES, IN_CHANNELS)


@pytest.mark.parametrize('conv_type', [LinAggrConv, AggrLinConv, ProjAggrConv])
def test_aggregation_convs_output_shape(graph, conv_type):
    x, _, edge_index = graph
    conv = conv_type(IN_CHANNELS, OUT_CHANNELS, norm=None)
    assert conv(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)
    assert conv.out_channels == OUT_CHANNELS


def test_proj_aggr_proj_conv_output_shape(graph):
    x, _, edge_index = graph
    conv = ProjAggrProjConv(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, norm=None)
    assert conv(x, edge_index).shape == (NUM_NODES, OUT_CHANNELS)
    assert conv.out_channels == OUT_CHANNELS


def test_mlp_aggr_conv_output_shape(graph):
    x, _, edge_index = graph
    batch = torch.zeros(NUM_NODES, dtype=torch.long)
    conv = MLPAggrConv(IN_CHANNELS, OUT_CHANNELS, norm=None)
    assert conv(x, batch, edge_index).shape == (NUM_NODES, OUT_CHANNELS)


def test_selection_conv_gathers_selected_nodes(graph):
    x, _, _ = graph
    selection = torch.tensor([1, 3, 5])
    assert torch.equal(SelectionConv(IN_CHANNELS)(x, selection), x[selection])


def test_proj_selection_conv_downsampling_shape(graph):
    x, _, _ = graph
    selection = torch.tensor([1, 3, 5])
    conv = ProjSelectionConv(IN_CHANNELS, OUT_CHANNELS, norm=None)
    assert conv(x, selection).shape == (3, OUT_CHANNELS)


def test_proj_selection_conv_upsampling_shape(graph):
    x, _, _ = graph
    selection = torch.tensor([1, 3, 5])
    conv = ProjSelectionConv(IN_CHANNELS, OUT_CHANNELS, norm=None)
    assert conv(x, selection, size=(3, NUM_NODES)).shape == (3, OUT_CHANNELS)


@pytest.mark.parametrize('edge_kind', ['pairwise', 'source'])
def test_simple_pointnet_conv_handles_both_edge_conventions(graph, edge_kind):
    x, pos, edge_index = graph
    edges = edge_index if edge_kind == 'pairwise' else _source_indexed()
    conv = SimplePointNetConv(IN_CHANNELS, OUT_CHANNELS, norm=None)
    assert conv(x, pos, edges).shape == (NUM_NODES, OUT_CHANNELS)
