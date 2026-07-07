import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.apply_selection import ApplySelection, select_knn_edges


@pytest.mark.parametrize(
    ('edge_index', 'selection_index', 'expected'),
    [
        pytest.param(
            # 4 nodes, k=2 neighbors each; keep nodes 0 and 2, and their
            # neighbors all survive the selection.
            torch.tensor([
                [0, 2],  # node 0 -> neighbors 0, 2
                [1, 3],  # node 1 (dropped)
                [2, 0],  # node 2 -> neighbors 2, 0
                [3, 1],  # node 3 (dropped)
            ]),
            torch.tensor([True, False, True, False]),
            torch.tensor([[0, 1], [1, 0]]),
            id='boolean_mask_no_drops',
        ),
        pytest.param(
            # 3 nodes, k=2; keep nodes 0 and 1. Node 0's second neighbor
            # (node 2) is dropped and should become a self-loop.
            torch.tensor([
                [0, 2],
                [1, 0],
                [2, 1],
            ]),
            torch.tensor([True, True, False]),
            torch.tensor([[0, 0], [1, 0]]),
            id='boolean_mask_dropped_neighbor_becomes_self_loop',
        ),
        pytest.param(
            # Same graph as above, but selection given as an integer index
            # tensor instead of a boolean mask.
            torch.tensor([
                [0, 2],
                [1, 0],
                [2, 1],
            ]),
            torch.tensor([0, 1]),
            torch.tensor([[0, 0], [1, 0]]),
            id='integer_index_selection',
        ),
        pytest.param(
            # Node 0's only neighbor (node 1) is dropped -> self-loop.
            torch.tensor([[1], [0]]),
            torch.tensor([True, False]),
            torch.tensor([[0]]),
            id='all_neighbors_dropped_become_self_loops',
        ),
    ],
)
def test_select_knn_edges(edge_index, selection_index, expected):
    out = select_knn_edges(edge_index, selection_index)
    assert torch.equal(out, expected)


def test_select_knn_edges_output_shape():
    edge_index = torch.randint(0, 10, (10, 4))
    mask = torch.zeros(10, dtype=torch.bool)
    mask[:6] = True

    out = select_knn_edges(edge_index, mask)

    assert out.shape == (6, 4)



@pytest.fixture
def make_data():
    """Factory fixture that builds a small `Data` object for testing.

    `edge_index` can either be standard PyG COO format `(2, E)` (the format
    PyG's own `is_edge_attr`/`is_node_attr` heuristics expect) or the dense
    `(N, K)` kNN format that `select_knn_edges` operates on.
    """

    def _make_data(num_nodes=4, dense_knn_edges=True, with_edge_attr=False, with_batch=False):
        x = torch.arange(num_nodes * 3, dtype=torch.float).reshape(num_nodes, 3)
        pos = torch.arange(num_nodes * 2, dtype=torch.float).reshape(num_nodes, 2)
        selection_index = torch.tensor([True, False, True, False][:num_nodes])

        if dense_knn_edges:
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=1)
            data = Data(x=x, pos=pos, edge_index=edge_index)
        else:
            # Standard COO edge_index with a number of edges distinct from
            # the number of nodes, so PyG correctly identifies edge- vs.
            # node-level attributes.
            num_edges = num_nodes + 2
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            data = Data(x=x, pos=pos, edge_index=edge_index)
            if with_edge_attr:
                data.edge_attr = torch.rand(num_edges, 2)

        data.selection_index = selection_index
        data.cluster_index = torch.arange(num_nodes)

        if with_batch:
            data.batch = torch.zeros(num_nodes, dtype=torch.long)
            data.ptr = torch.tensor([0, num_nodes])

        return data

    return _make_data

class TestApplySelection:
    def test_passthrough_without_selection_index(self):
        data = Data(x=torch.randn(3, 2))
        transform = ApplySelection()

        out = transform(data)

        # BaseTransform.__call__ shallow-copies the Data wrapper, so `out` won't
        # be the same object, but it should carry the same tensors untouched.
        assert out.x is data.x


    def test_node_attrs_filtered(self, make_data):
        data = make_data(num_nodes=4)
        transform = ApplySelection()

        out = transform(data)

        assert torch.equal(out.x, data.x[data.selection_index])
        assert torch.equal(out.pos, data.pos[data.selection_index])
        assert out.num_nodes == 2


    def test_edge_index_remapped(self, make_data):
        # Only meaningful for the dense (N, K) kNN edge_index format select_knn_edges expects.
        data = make_data(num_nodes=4, dense_knn_edges=True)
        transform = ApplySelection()

        out = transform(data)

        assert 'edge_index' in out
        assert out.edge_index.shape == (2, 2)


    def test_assertion_on_empty_selection(self, make_data):
        data = make_data(num_nodes=4)
        data.selection_index = torch.tensor([], dtype=torch.bool)
        transform = ApplySelection()

        with pytest.raises(ValueError):
            transform(data)


    def test_ptr_recomputed_from_batch(self, make_data):
        data = make_data(num_nodes=4, with_batch=True)
        data.batch = torch.tensor([0, 0, 1, 1])
        data.selection_index = torch.tensor([True, False, True, True])
        transform = ApplySelection()

        out = transform(data)

        # Kept nodes 0, 2, 3 -> batch becomes [0, 1, 1] -> counts [1, 2]
        assert torch.equal(out.batch, torch.tensor([0, 1, 1]))
        assert torch.equal(out.ptr, torch.tensor([0, 1, 3]))


