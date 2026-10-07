import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.apply_selection import ApplySelection, select_knn_edges


@pytest.mark.parametrize(
    ('edge_index', 'selection_index', 'expected'),
    [
        pytest.param(
            torch.tensor([[0, 2], [1, 3], [2, 0], [3, 1]]),
            torch.tensor([True, False, True, False]),
            torch.tensor([[0, 1], [1, 0]]),
            id='boolean_mask_no_drops',
        ),
        pytest.param(
            torch.tensor([[0, 2], [1, 0], [2, 1]]),
            torch.tensor([True, True, False]),
            torch.tensor([[0, 0], [1, 0]]),
            id='dropped_neighbor_becomes_self_loop',
        ),
        pytest.param(
            torch.tensor([[0, 2], [1, 0], [2, 1]]),
            torch.tensor([0, 1]),
            torch.tensor([[0, 0], [1, 0]]),
            id='integer_index_selection',
        ),
        pytest.param(
            torch.tensor([[1], [0]]),
            torch.tensor([True, False]),
            torch.tensor([[0]]),
            id='all_neighbors_dropped_become_self_loops',
        ),
    ],
)
def test_select_knn_edges(edge_index, selection_index, expected):
    assert torch.equal(select_knn_edges(edge_index, selection_index), expected)


def test_select_knn_edges_output_shape():
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:6] = True

    assert select_knn_edges(torch.randint(0, 10, (10, 4)), mask).shape == (6, 4)


@pytest.fixture
def data():
    return Data(
        x=torch.arange(12, dtype=torch.float).reshape(4, 3),
        pos=torch.arange(8, dtype=torch.float).reshape(4, 2),
        edge_index=torch.stack([torch.arange(4), torch.arange(4)], dim=1),
        selection_index=torch.tensor([True, False, True, False]),
        cluster_index=torch.arange(4),
    )


def test_passthrough_without_selection_index():
    data = Data(x=torch.randn(3, 2))

    assert ApplySelection()(data).x is data.x


def test_node_attrs_filtered(data):
    out = ApplySelection()(data)

    assert torch.equal(out.x, data.x[data.selection_index])
    assert torch.equal(out.pos, data.pos[data.selection_index])
    assert out.num_nodes == 2


def test_edge_index_remapped(data):
    assert ApplySelection()(data).edge_index.shape == (2, 2)


def test_assertion_on_empty_selection(data):
    data.selection_index = torch.tensor([], dtype=torch.bool)

    with pytest.raises(ValueError):
        ApplySelection()(data)


def test_ptr_recomputed_from_batch():
    data = Data(
        x=torch.arange(12, dtype=torch.float).reshape(4, 3),
        pos=torch.arange(8, dtype=torch.float).reshape(4, 2),
        edge_index=torch.stack([torch.arange(4), torch.arange(4)], dim=1),
        batch=torch.tensor([0, 0, 1, 1]),
        selection_index=torch.tensor([True, False, True, True]),
    )

    out = ApplySelection()(data)

    # Kept nodes 0, 2, 3 -> counts [1, 2]
    assert torch.equal(out.batch, torch.tensor([0, 1, 1]))
    assert torch.equal(out.ptr, torch.tensor([0, 1, 3]))
