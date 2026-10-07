import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

from clouds.transforms.ball import BallGraph


@pytest.fixture
def data():
    return Data(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
    )


def test_forward_basic(data):
    edge_index = BallGraph(r=1.2, max_num_neighbors=5)(data).edge_index

    assert edge_index.shape[0] == 2
    for source, dest in edge_index.T:
        assert torch.linalg.vector_norm(data.pos[source] - data.pos[dest]) < 1.2


def test_forward_small_radius(data):
    # Only self-edges should be created.
    assert BallGraph(r=0.1, max_num_neighbors=5)(data).edge_index.shape[1] == data.num_nodes


def test_forward_large_radius(data):
    edge_index = BallGraph(r=3.0, max_num_neighbors=5)(data).edge_index

    assert 0 < edge_index.shape[1] <= 5 * 5


def test_forward_preserves_original_data(data):
    original_pos = data.pos.clone()

    result = BallGraph(r=1.5, max_num_neighbors=5)(data)

    assert torch.allclose(original_pos, result.pos)
    assert result.pos.shape == data.pos.shape


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_forward_with_batch():
    batch = torch.tensor([0, 0, 1, 1])
    data = Data(
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]]),
        batch=batch,
    )

    edge_index = BallGraph(r=1.5, max_num_neighbors=2)(data).edge_index

    for source, dest in edge_index.T:
        assert batch[source] == batch[dest]


def test_forward_asserts_pos_tensor():
    with pytest.raises(AssertionError):
        BallGraph(r=1.0)(Data())


@pytest.mark.parametrize(
    'radius,expected_min_edges',
    [(0.5, 0), (1.5, 2), (3.0, 4)],
)
def test_forward_different_radii(data, radius, expected_min_edges):
    assert BallGraph(r=radius, max_num_neighbors=5)(data).edge_index.shape[1] >= expected_min_edges
