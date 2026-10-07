from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.cut_select import CutSelect


@pytest.fixture
def data():
    return Data(pos=torch.randn(100, 3), num_nodes=100)


def test_init_default_parameters():
    cut = CutSelect(max_num_points=10)

    assert cut.max_num_points == 10
    assert cut.max_ratio == 1.0
    assert cut.sort_by_distance is False
    assert cut.dims is None


def test_init_custom_parameters():
    cut = CutSelect(max_num_points=20, max_ratio=0.5, sort_by_distance=True, dims=[0, 1])

    assert cut.max_num_points == 20
    assert cut.max_ratio == 0.5
    assert cut.sort_by_distance is True
    assert cut.dims == [0, 1]


@pytest.mark.parametrize(
    'kwargs,expected',
    [
        (
            {'max_num_points': 20, 'max_ratio': 0.5, 'sort_by_distance': True, 'dims': [0, 1]},
            'CutSelect(max_num_points=20, max_ratio=0.5, sort_by_distance=True, dims=[0, 1])',
        ),
        ({'max_num_points': 10}, 'CutSelect(max_num_points=10, max_ratio=1.0, sort_by_distance=False, dims=None)'),
    ],
)
def test_repr(kwargs, expected):
    assert repr(CutSelect(**kwargs)) == expected


@pytest.mark.parametrize(
    'max_num_points,max_ratio,expected_count',
    [
        (10, 1.0, 10),
        (100, 0.3, 30),
        (5, 0.8, 5),
        (15, 0.5, 15),
        (200, 0.3, 30),
        (200, 1.0, 100),
        (50, 0.2, 20),
        (0, 1.0, 0),
        (100, 0.0, 0),
    ],
)
def test_forward_count(data, max_num_points, max_ratio, expected_count):
    assert len(CutSelect(max_num_points=max_num_points, max_ratio=max_ratio)(data).selection_index) == expected_count


def test_forward_selection_index_properties(data):
    selection_index = CutSelect(max_num_points=10)(data).selection_index

    assert selection_index.device == data.pos.device
    assert selection_index.dtype == torch.long
    assert torch.all(selection_index >= 0)
    assert torch.all(selection_index < data.num_nodes)


def test_forward_with_sort_by_distance(data):
    assert len(CutSelect(max_num_points=10, sort_by_distance=True)(data).selection_index) == 10


def test_forward_with_dims(data):
    assert len(CutSelect(max_num_points=10, dims=[0, 1])(data).selection_index) == 10


def test_forward_with_single_dim():
    data = Data(pos=torch.randn(100, 2), num_nodes=100)

    assert len(CutSelect(max_num_points=10, dims=[0])(data).selection_index) == 10


@pytest.mark.parametrize('dim', [2, 3, 128])
def test_forward_dimensions(dim):
    data = Data(pos=torch.randn(50, dim), num_nodes=50)

    assert len(CutSelect(max_num_points=10)(data).selection_index) == 10


def test_forward_keeps_original_data(data):
    original_pos = data.pos.clone()

    assert torch.equal(data.pos, original_pos)
    assert data.num_nodes == 100
    assert CutSelect(max_num_points=10)(data).pos is data.pos


def test_forward_invalid_batch():
    data = Data(pos=torch.randn(10, 3), num_nodes=10, batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))

    with pytest.raises(AssertionError):
        CutSelect(max_num_points=5)(data)


def test_forward_invalid_pos_type():
    with pytest.raises(AssertionError):
        CutSelect(max_num_points=5)(Data(pos=[1, 2, 3], num_nodes=3))


@patch('torch.randn')
def test_forward_vector_randomness(mock_randn, data):
    mock_randn.return_value = torch.tensor([1.0, 0.0, 0.0])

    CutSelect(max_num_points=10)(data)

    mock_randn.assert_called_once_with([3], device=data.pos.device)


def test_forward_consistency_with_same_seed(data):
    torch.manual_seed(42)
    first = CutSelect(max_num_points=10)(data).selection_index

    torch.manual_seed(42)
    second = CutSelect(max_num_points=10)(data).selection_index

    assert torch.equal(first, second)
