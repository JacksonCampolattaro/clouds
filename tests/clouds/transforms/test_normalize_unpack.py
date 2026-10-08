import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms import ClampPos, NormalizeScale, UnpackSourceGraph


def test_clamp_pos_bounds_positions():
    data = Data(pos=torch.randn(5, 3) * 10)

    out = ClampPos([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])(data)

    assert out.pos.abs().max().item() <= 1.0 + 1e-6


def test_normalize_scale_normalizes_the_tail_of_the_distribution():
    out = NormalizeScale()(Data(pos=torch.randn(50, 3) * 10))

    quantile = torch.quantile(out.pos.abs().flatten(), 0.99).item()
    assert quantile == pytest.approx(0.999999, rel=1e-3)


def test_unpack_source_graph_converts_to_pairwise_edges():
    data = Data(pos=torch.randn(3, 3), edge_index=torch.tensor([[1, 2], [2, 0], [0, 1]]))

    out = UnpackSourceGraph()(data)

    assert out.edge_index.shape == (2, 6)
    # First half of the flattened source-index edges are the sources
    assert torch.equal(out.edge_index[0], torch.tensor([1, 2, 2, 0, 0, 1]))
    assert torch.equal(out.edge_index[1], torch.tensor([0, 0, 1, 1, 2, 2]))
