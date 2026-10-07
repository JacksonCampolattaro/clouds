import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.block_select import BlockSelect


@pytest.fixture
def basic_data():
    return Data(pos=torch.randn(100, 3), num_nodes=100)


@pytest.fixture
def batched_data():
    return Data(
        pos=torch.randn(100, 3),
        batch=torch.tensor([0] * 50 + [1] * 30 + [2] * 20),
        ptr=torch.tensor([0, 50, 80, 100]),
        num_nodes=100,
    )


def test_init_defaults():
    transform = BlockSelect()
    assert transform.max_num_points == int(1e9)
    assert transform.min_num_points == 1
    assert transform.selection_factor == 1.0


def test_init_custom_params():
    transform = BlockSelect(max_num_points=1000, min_num_points=10, selection_factor=0.5)
    assert transform.max_num_points == 1000
    assert transform.min_num_points == 10
    assert transform.selection_factor == 0.5


def test_invalid_parameters():
    # No validation in the constructor: values are only clamped during forward.
    transform = BlockSelect(max_num_points=-1, min_num_points=-5, selection_factor=-1.0)
    assert transform.max_num_points == -1
    assert transform.min_num_points == -5
    assert transform.selection_factor == -1.0


def test_repr():
    repr_str = repr(BlockSelect(max_num_points=1000, selection_factor=0.5))
    assert 'BlockSelect' in repr_str
    assert '*0.5' in repr_str
    assert '<1000' in repr_str


def test_data_attributes_preserved(basic_data):
    result = BlockSelect(selection_factor=0.3)(basic_data)

    assert result.pos is basic_data.pos
    assert result.num_nodes == basic_data.num_nodes
    assert hasattr(result, 'selection_index')


def test_large_number_points():
    data = Data(pos=torch.randn(10000, 3), num_nodes=10000)

    assert len(BlockSelect(max_num_points=5000, selection_factor=0.8)(data).selection_index) == 5000


@pytest.mark.parametrize(
    'selection_factor,min_points,max_points,expected_size',
    [
        (0.0, 1, 100, 1),
        (0.3, 1, 100, 30),
        (0.5, 1, 100, 50),
        (1.0, 1, 100, 100),
        (2.0, 1, 100, 100),
        (0.5, 30, 100, 50),
        (0.1, 30, 100, 30),
        (0.9, 30, 50, 50),
    ],
)
def test_selection_size_basic(basic_data, selection_factor, min_points, max_points, expected_size):
    selection_index = BlockSelect(selection_factor=selection_factor, min_num_points=min_points, max_num_points=max_points)(
        basic_data
    ).selection_index

    assert torch.equal(selection_index, torch.arange(expected_size))


@pytest.mark.parametrize(
    'selection_factor,min_points,max_points,expected_sizes',
    [
        (0.0, 1, 100, [1, 1, 1]),
        (0.5, 1, 100, [25, 15, 10]),
        (1.0, 1, 100, [50, 30, 20]),
        (0.5, 20, 100, [25, 20, 20]),
        (0.9, 10, 15, [15, 15, 15]),
    ],
)
def test_selection_size_batched(batched_data, selection_factor, min_points, max_points, expected_sizes):
    result = BlockSelect(selection_factor=selection_factor, min_num_points=min_points, max_num_points=max_points)(batched_data)

    offset = 0
    for i, size in enumerate(expected_sizes):
        start = batched_data.ptr[i].item()
        assert torch.equal(result.selection_index[offset : offset + size], torch.arange(start, start + size))
        offset += size

    assert len(result.selection_index) == sum(expected_sizes)
