import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from clouds.data import MultiGridData
from clouds.transforms import BuildPhantomMultigrid, FlattenMultigrid, KNNUpsampleNeighbors, MultiGridTransform


def _level_data(num_nodes: int = 6) -> Data:
    return Data(pos=torch.randn(num_nodes, 3), x=torch.randn(num_nodes, 2))


class _AddOne(BaseTransform):
    def forward(self, data: Data) -> Data:
        data.pos = data.pos + 1.0
        return data


def test_multi_grid_transform_applies_one_transform_per_level():
    mg = MultiGridData()
    mg.set_level('x0', Data(pos=torch.zeros(6, 3)))
    mg.set_level('x1', Data(pos=torch.zeros(3, 3)))

    out = MultiGridTransform(_AddOne, levels=[0, 1])(mg)

    assert set(out.node_types) == {'x0', 'x1'}
    assert torch.equal(out.get_level('x0').pos, torch.ones(6, 3))
    assert torch.equal(out.get_level('x1').pos, torch.ones(3, 3))


def test_build_phantom_multigrid_places_data_on_coarse_level():
    data = _level_data(5)

    out = BuildPhantomMultigrid()(data)

    assert isinstance(out, MultiGridData)
    assert out.node_types == ['x0', 'x1']
    assert out.get_level('x0').num_nodes in (None, 0)
    assert torch.equal(out.get_level('x1').pos, data.pos)


def test_knn_upsample_neighbors_connects_levels():
    mg = MultiGridData()
    mg.set_level('x0', _level_data(8))
    mg.set_level('x1', _level_data(3))

    out = KNNUpsampleNeighbors(k=2)(mg)

    # Neighbors are indexed by the query points (fine level x0).
    assert out['x1', 'to', 'x0'].edge_index.shape == (8, 2)


def test_flatten_multigrid_passthrough_for_flat_data():
    data = Data(pos=torch.randn(4, 3))

    assert torch.equal(FlattenMultigrid()(data).pos, data.pos)


def test_flatten_multigrid_concatenates_levels():
    mg = MultiGridData()
    mg.set_level('x0', _level_data(4))
    mg.set_level('x1', _level_data(2))

    out = FlattenMultigrid()(mg)

    assert out.num_nodes == 6
    assert out.level.shape == (6, 1)
    assert set(out.level.flatten().tolist()) == {0.0, 1.0}
