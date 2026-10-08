"""Smoke tests for the graph convolution layers on tiny synthetic graphs.

These check channel bookkeeping and both edge conventions used in this
codebase: PyG-style pairwise ``(2, E)`` indices and source-indexed ``(N, K)``
indices. No datasets are required.
"""

import pytest
import torch
from torch_geometric.data import Data

from clouds.nn.apply import apply_to_data
from clouds.nn.conv.aggr import (
    AggrConv,
    AggrLinConv,
    LinAggrConv,
    MLPAggrConv,
    ProjAggrConv,
    ProjAggrProjConv,
)
from clouds.nn.conv.dela import DeLABlock, DeLAConv, DeLAMLPConv, SimpleDeLABlock
from clouds.nn.conv.next import InvResConv
from clouds.nn.conv.pointnet import SimplePointNetConv
from clouds.nn.conv.selection import ProjSelectionConv, SelectionConv
from clouds.nn.conv.spatial_encoding import RelativePositionRegularizer, SpatialEncodingConv, SplitSpatialEncodingConv
from clouds.nn.norm import DeLABatchNorm

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


@pytest.mark.parametrize('edge_kind', ['pairwise', 'source'])
def test_dela_conv_handles_both_edge_conventions(graph, edge_kind):
    x, _, edge_index = graph
    edges = edge_index if edge_kind == 'pairwise' else _source_indexed()
    assert DeLAConv(IN_CHANNELS, OUT_CHANNELS)(x, edges).shape == (NUM_NODES, OUT_CHANNELS)


def test_inv_res_conv_output_shape(graph):
    x, pos, edge_index = graph
    conv = InvResConv(IN_CHANNELS, IN_CHANNELS, act='relu')
    assert conv(x, pos, edge_index).shape == (NUM_NODES, IN_CHANNELS)


def test_relative_position_regularizer_returns_scalar_loss(graph):
    x, pos, edge_index = graph
    loss = RelativePositionRegularizer(IN_CHANNELS)(x, pos, edge_index)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_de_la_mlp_conv_runs_through_apply_to_data(graph):
    x, _, edge_index = graph
    data = Data(x=x.clone(), edge_index=edge_index)
    # norm=None avoids the BatchNorm/init_weight mismatch in this standalone setup.
    out = apply_to_data(DeLAMLPConv(IN_CHANNELS, out_channels=OUT_CHANNELS, norm=None), data)
    assert out.x.shape == (NUM_NODES, OUT_CHANNELS)


@pytest.mark.parametrize('block_type', [SimpleDeLABlock, DeLABlock])
def test_dela_blocks_run_through_apply_to_data(graph, block_type):
    x, _, edge_index = graph
    data = Data(x=x.clone(), edge_index=edge_index)
    block = block_type(IN_CHANNELS, out_channels=OUT_CHANNELS, num_convs_per_block=1, norm=None)
    out = apply_to_data(block, data)
    assert out.x.shape == (NUM_NODES, OUT_CHANNELS)


def test_spatial_encoding_conv_accepts_none_norm_kwargs(graph):
    # Regression: norm_kwargs=None was treated as a dict and caused
    # `dict | None`, breaking configs.shapenet.mg_dela during construction.
    x, pos, edge_index = graph
    conv = SpatialEncodingConv(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        hidden_channels=48,
        init_weight=0.8,
        norm=DeLABatchNorm,
        norm_kwargs=None,
    )
    assert conv(x, pos, edge_index).shape == (NUM_NODES, OUT_CHANNELS)


def test_split_spatial_encoding_conv_accepts_none_norm_kwargs(graph):
    x, pos, edge_index = graph
    conv = SplitSpatialEncodingConv(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        hidden_channels=48,
        init_weight=0.8,
        norm=DeLABatchNorm,
        norm_kwargs=None,
    )
    assert conv(x, pos, edge_index).shape == (NUM_NODES, OUT_CHANNELS)


def test_dela_block_accepts_none_norm_kwargs(graph):
    x, _, edge_index = graph
    data = Data(x=x.clone(), edge_index=edge_index)
    block = DeLABlock(
        IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_convs_per_block=1,
        norm=DeLABatchNorm,
        norm_kwargs=None,
    )
    assert apply_to_data(block, data).x.shape == (NUM_NODES, OUT_CHANNELS)
