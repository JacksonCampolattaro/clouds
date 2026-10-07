import math

import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.radius_select import RadiusSelect


@pytest.fixture
def data():
    return Data(pos=torch.tensor([[float(i), 0.0] for i in range(10)]))


def test_init_and_repr():
    repr_str = repr(RadiusSelect(max_num_points=5, max_radius=2.0, max_ratio=0.5))

    assert 'RadiusSelect' in repr_str
    assert 'max_num_points=5' in repr_str
    assert 'max_radius=2.0' in repr_str
    assert 'max_ratio=0.5' in repr_str
    assert 'sort_by_distance=False' in repr_str
    assert 'deterministic=False' in repr_str
    assert 'dims=None' in repr_str


def test_selection_with_deterministic(data):
    # deterministic=True centers on pos[0], so the closest points are 0, 1, 2.
    assert RadiusSelect(max_num_points=3, deterministic=True)(data).selection_index.tolist() == [0, 1, 2]


def test_selection_with_sort_by_distance(data):
    assert RadiusSelect(max_num_points=3, deterministic=True, sort_by_distance=True)(data).selection_index.tolist() == [0, 1, 2]


def test_selection_without_sort_by_distance(data):
    result = RadiusSelect(max_num_points=5, deterministic=True, sort_by_distance=False)(data)

    assert result.selection_index.tolist() == sorted(result.selection_index.tolist())


def test_radius_filtering(data):
    # Within radius 2.5 of point 0 are indices 0, 1, 2.
    assert RadiusSelect(max_num_points=4, max_radius=2.5, deterministic=True, sort_by_distance=True)(
        data
    ).selection_index.tolist() == [0, 1, 2]


def test_max_ratio_limiting(data):
    assert RadiusSelect(max_num_points=10, max_ratio=0.3, deterministic=True, sort_by_distance=True)(
        data
    ).selection_index.tolist() == [0, 1, 2]


def test_dims_selection():
    data = Data(pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))

    assert RadiusSelect(max_num_points=2, deterministic=True, dims=[0, 1], sort_by_distance=True)(
        data
    ).selection_index.tolist() == [0, 1]


def test_empty_selection_when_no_points_within_radius(data):
    assert RadiusSelect(max_num_points=10, max_radius=0.1, deterministic=True, sort_by_distance=True)(
        data
    ).selection_index.tolist() == [0]


def test_math_inf_radius(data):
    assert RadiusSelect(max_num_points=5, max_radius=math.inf, deterministic=True, sort_by_distance=True)(
        data
    ).selection_index.tolist() == [0, 1, 2, 3, 4]
