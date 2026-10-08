import torch
from torch_geometric.data import Data

from clouds.transforms import CylinderSample, CylinderSelect, SlabSample, SlabSelect


def _cloud(num_nodes: int = 100) -> Data:
    return Data(pos=torch.randn(num_nodes, 3))


def test_cylinder_select_marks_at_most_max_num_points():
    data = _cloud(100)

    out = CylinderSelect(max_num_points=10, deterministic=True)(data)

    assert out.selection_index.numel() <= 10


def test_cylinder_select_keeps_points_within_radius():
    data = Data(pos=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.2, 0.0]]))

    out = CylinderSelect(max_num_points=4, max_radius=1.0, deterministic=True)(data)

    assert torch.all(out.pos[out.selection_index][:, :2].norm(dim=-1) < 1.0)


def test_cylinder_sample_filters_nodes():
    data = _cloud(100)

    out = CylinderSample(max_num_points=10, deterministic=True)(data)

    assert out.num_nodes <= 10
    assert 'selection_index' not in out


def test_slab_select_marks_at_most_max_num_points():
    data = _cloud(100)

    out = SlabSelect(max_num_points=10, deterministic=True)(data)

    assert out.selection_index.numel() <= 10


def test_slab_sample_filters_nodes():
    data = _cloud(100)

    out = SlabSample(max_num_points=10, deterministic=True)(data)

    assert out.num_nodes <= 10
    assert 'selection_index' not in out
