import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.random_select import RandomSelect


@pytest.fixture
def simple_data():
    return Data(pos=torch.randn(10, 3), x=torch.randn(10, 5), num_nodes=10)


@pytest.fixture
def batched_data():
    return Data(
        pos=torch.randn(15, 3),
        x=torch.randn(15, 5),
        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        ptr=torch.tensor([0, 5, 10, 15]),
        num_nodes=15,
    )


@pytest.fixture
def batched_data_uneven():
    return Data(
        pos=torch.randn(12, 3),
        x=torch.randn(12, 5),
        batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]),
        ptr=torch.tensor([0, 4, 7, 12]),
        num_nodes=12,
    )


def test_init_default():
    transform = RandomSelect()

    assert transform.max_num_points == 1e7
    assert transform.min_num_points == 1
    assert transform.selection_factor == 1.0
    assert transform.replacement is False


def test_init_custom():
    transform = RandomSelect(max_num_points=100, min_num_points=5, selection_factor=0.5, replacement=True)

    assert transform.max_num_points == 100
    assert transform.min_num_points == 5
    assert transform.selection_factor == 0.5
    assert transform.replacement is True


def test_init_tuple_selection_factor():
    transform = RandomSelect(selection_factor=(0.2, 0.8))

    assert isinstance(transform.selection_factor, tuple)
    assert len(transform.selection_factor) == 2


def test_forward_constant_factor(simple_data):
    result = RandomSelect(selection_factor=0.5, max_num_points=100, min_num_points=1)(simple_data)

    assert len(result.selection_index) == 5
    assert result.selection_index.dtype == torch.long
    assert torch.all(result.selection_index >= 0)
    assert torch.all(result.selection_index < simple_data.num_nodes)
    assert torch.all(result.selection_index == result.selection_index.sort()[0])


def test_forward_min_num_points(simple_data):
    result = RandomSelect(selection_factor=0.1, min_num_points=3, max_num_points=100)(simple_data)

    assert 3 <= len(result.selection_index) <= simple_data.num_nodes


def test_forward_max_num_points(simple_data):
    result = RandomSelect(selection_factor=2.0, max_num_points=8, min_num_points=1)(simple_data)

    assert len(result.selection_index) <= 8
    assert len(result.selection_index) <= simple_data.num_nodes


def test_forward_with_replacement_single_graph(simple_data):
    result = RandomSelect(selection_factor=1.5, replacement=True, max_num_points=100)(simple_data)

    assert len(result.selection_index) == 15
    assert torch.all(result.selection_index >= 0)
    assert torch.all(result.selection_index < simple_data.num_nodes)
    assert len(torch.unique(result.selection_index)) < len(result.selection_index)


def test_forward_with_replacement_batched(batched_data):
    result = RandomSelect(selection_factor=2, replacement=True, max_num_points=100)(batched_data)

    assert len(result.selection_index) == 30
    for start, end in zip([0, 5, 10], [5, 10, 15], strict=True):
        assert ((result.selection_index >= start) & (result.selection_index < end)).sum().item() == 10


def test_forward_with_replacement_batched_uneven(batched_data_uneven):
    result = RandomSelect(selection_factor=0.8, replacement=True, max_num_points=100)(batched_data_uneven)

    for (start, end), expected in zip(zip([0, 4, 7], [4, 7, 12], strict=True), [3, 2, 4], strict=True):
        assert ((result.selection_index >= start) & (result.selection_index < end)).sum().item() == expected


def test_forward_with_replacement_and_max_limit(simple_data):
    result = RandomSelect(selection_factor=5.0, replacement=True, max_num_points=20)(simple_data)

    assert len(result.selection_index) == 20


def test_forward_batched_data_without_replacement(batched_data):
    result = RandomSelect(selection_factor=0.6, max_num_points=100, min_num_points=1)(batched_data)

    assert len(result.selection_index) == 9
    assert len(torch.unique(result.selection_index)) == len(result.selection_index)
    for start, end in zip([0, 5, 10], [5, 10, 15], strict=True):
        assert ((result.selection_index >= start) & (result.selection_index < end)).sum().item() == 3


def test_forward_batched_data_min_num_points(batched_data):
    result = RandomSelect(selection_factor=0.1, min_num_points=2, max_num_points=100)(batched_data)

    for start, end in zip([0, 5, 10], [5, 10, 15], strict=True):
        assert ((result.selection_index >= start) & (result.selection_index < end)).sum().item() >= 2


def test_forward_batched_data_max_num_points(batched_data):
    result = RandomSelect(selection_factor=2.0, max_num_points=3, min_num_points=1)(batched_data)

    assert len(result.selection_index) == 9


def test_forward_with_tuple_selection_factor(simple_data):
    transform = RandomSelect(selection_factor=(0.3, 0.7), max_num_points=100, min_num_points=1)

    counts = [len(transform(simple_data).selection_index) for _ in range(10)]

    assert len(set(counts)) > 1
    assert all(3 <= count <= 7 for count in counts)


def test_forward_preserves_other_attributes(simple_data):
    original_x = simple_data.x.clone()
    original_pos = simple_data.pos.clone()

    result = RandomSelect(selection_factor=0.5)(simple_data)

    assert torch.equal(result.x, original_x)
    assert torch.equal(result.pos, original_pos)
    assert result.num_nodes == simple_data.num_nodes


@pytest.mark.parametrize('replacement', [False, True])
def test_selection_index_sorted(simple_data, replacement):
    result = RandomSelect(selection_factor=0.5, replacement=replacement)(simple_data)

    assert torch.all(result.selection_index == result.selection_index.sort()[0])


def test_edge_case_single_node():
    data = Data(pos=torch.randn(1, 3), x=torch.randn(1, 5), num_nodes=1)

    assert len(RandomSelect(selection_factor=1.0, min_num_points=1, max_num_points=100)(data).selection_index) == 1
    assert torch.all(RandomSelect(selection_factor=3.0, replacement=True, max_num_points=100)(data).selection_index == 0)


def test_edge_case_zero_selection():
    data = Data(pos=torch.randn(10, 3), num_nodes=10)

    assert len(RandomSelect(selection_factor=0.0, min_num_points=0, max_num_points=100)(data).selection_index) == 0
    assert len(RandomSelect(selection_factor=0.0, replacement=True, min_num_points=0)(data).selection_index) == 0


def test_batched_replacement_with_zero_selection(batched_data):
    result = RandomSelect(selection_factor=0.0, replacement=True, min_num_points=0, max_num_points=100)(batched_data)

    assert len(result.selection_index) == 0
