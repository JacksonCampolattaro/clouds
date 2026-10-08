"""Tests for MultiGridData, the multi-level heterogeneous container."""

import pytest
import torch
from torch_geometric.data import Data

from clouds.data import MultiGridData


def _level(num_nodes: int = 4, in_channels: int = 3) -> Data:
    edge_index = torch.stack([torch.arange(num_nodes - 1), torch.arange(1, num_nodes)])
    return Data(
        x=torch.randn(num_nodes, in_channels),
        pos=torch.randn(num_nodes, 3),
        edge_index=edge_index,
        edge_attr=torch.randn(num_nodes - 1, 2),
    )


def test_set_and_get_level_round_trip():
    mg = MultiGridData()
    level = _level()
    mg.set_level('x0', level)

    out = mg.get_level('x0')
    assert torch.equal(out.x, level.x)
    assert torch.equal(out.pos, level.pos)
    assert torch.equal(out.edge_index, level.edge_index)
    assert torch.equal(out.edge_attr, level.edge_attr)


def test_set_level_replaces_previous_contents():
    mg = MultiGridData()
    mg.set_level('x0', _level(4))
    mg.set_level('x0', _level(2))
    assert mg.get_level('x0').x.shape == (2, 3)


def test_get_level_accepts_integer_index():
    mg = MultiGridData()
    mg.set_level('x0', _level())
    assert torch.equal(mg.get_level(0).x, mg.get_level('x0').x)


def test_get_levels_selects_requested_levels():
    mg = MultiGridData()
    mg.set_level('x0', _level())
    mg.set_level('x1', _level(2))

    sub = mg.get_levels(['x1'])
    assert sub.node_types == ['x1']
    assert torch.equal(sub.get_level('x1').x, mg.get_level('x1').x)


def test_get_levels_rename_relabels_levels():
    mg = MultiGridData()
    mg.set_level('coarse', _level(2))

    renamed = mg.get_levels(['coarse'], rename=True)
    assert renamed.node_types == ['x0']
    assert torch.equal(renamed.get_level('x0').x, mg.get_level('coarse').x)


def test_get_levels_rejects_unknown_levels():
    mg = MultiGridData()
    mg.set_level('x0', _level())
    with pytest.raises(KeyError):
        mg.get_levels(['does-not-exist'])


def test_non_node_attributes_become_shared_globals():
    level = _level()
    level.name = 'demo'
    mg = MultiGridData()
    mg.set_level('x0', level)

    assert mg.get_globals()['name'] == 'demo'
    assert mg.get_level('x0').name == 'demo'


def test_cat_dim_keeps_dense_source_indexed_edges():
    mg = MultiGridData()
    mg.set_level('x0', Data(pos=torch.randn(4, 3), edge_index=torch.randint(0, 4, (4, 3))))
    store = mg['x0', 'to', 'x0']

    # Dense (N, K) edges must be concatenated along dim 0, not dim -1.
    assert mg.__cat_dim__('edge_index', store.edge_index, store) == 0


def test_inc_offsets_dense_edges_across_graphs():
    mg = MultiGridData()
    mg.set_level('x0', Data(pos=torch.randn(4, 3), edge_index=torch.randint(0, 4, (4, 3))))
    store = mg['x0', 'to', 'x0']

    # Dense (N, K) edges are shifted by their max index plus one.
    assert mg.__inc__('edge_index', store.edge_index, store).item() == store.edge_index.max().item() + 1


def test_inc_offsets_pairwise_edges_by_num_nodes():
    mg = MultiGridData()
    mg.set_level('x0', Data(pos=torch.randn(4, 3), edge_index=torch.stack([torch.arange(4), torch.arange(4)])))
    store = mg['x0', 'to', 'x0']

    assert torch.equal(mg.__inc__('edge_index', store.edge_index, store), torch.full((2, 1), 4, dtype=store.edge_index.dtype))
