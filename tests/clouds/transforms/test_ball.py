import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

from clouds.transforms.ball import BallGraph


class TestBallGraph:
    @pytest.fixture
    def simple_data(self):
        """Create a simple dataset with 5 points in 2D space."""
        pos = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=torch.float,
        )
        return Data(pos=pos)

    def test_forward_basic(self, simple_data):
        """Test forward pass with radius that captures some but not all points."""
        transform = BallGraph(r=1.2, max_num_neighbors=5)
        result = transform(simple_data)

        # Should have edge_index attribute
        assert hasattr(result, 'edge_index')
        assert result.edge_index.shape[0] == 2

        # Check that edges are within radius
        edge_index = result.edge_index
        for i in range(edge_index.shape[1]):
            source = edge_index[0, i]
            dest = edge_index[1, i]
            dist = torch.linalg.vector_norm(simple_data.pos[source] - simple_data.pos[dest])
            assert dist < 1.2

    def test_forward_small_radius(self, simple_data):
        """Test forward pass with very small radius (should produce no edges)."""
        transform = BallGraph(r=0.1, max_num_neighbors=5)
        result = transform(simple_data)

        # With small radius, only self-edges should be created
        assert result.edge_index.shape[1] == simple_data.num_nodes

    def test_forward_large_radius(self, simple_data):
        """Test forward pass with large radius (should connect all points)."""
        transform = BallGraph(r=3.0, max_num_neighbors=5)
        result = transform(simple_data)

        # With large radius, should have edges but limited by k
        assert result.edge_index.shape[1] > 0
        # Should not exceed max_num_neighbors * num_points
        assert result.edge_index.shape[1] <= 5 * 5  # 5 points * max_num_neighbors

    def test_forward_preserves_original_data(self, simple_data):
        """Test that original data attributes are preserved."""
        transform = BallGraph(r=1.5, max_num_neighbors=5)
        original_pos = simple_data.pos.clone()

        result = transform(simple_data)

        # Original pos should be unchanged
        assert torch.allclose(original_pos, result.pos)
        assert result.pos.shape == simple_data.pos.shape

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_forward_with_batch(self):
        """Test forward pass with batch information."""
        pos = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [10.0, 10.0],
                [11.0, 10.0],
            ],
            dtype=torch.float,
        )
        batch = torch.tensor([0, 0, 1, 1])
        data = Data(pos=pos, batch=batch)

        transform = BallGraph(r=1.5, max_num_neighbors=2)
        result = transform(data)

        # Should connect points within same batch that are close
        assert hasattr(result, 'edge_index')

        # Points from different batches shouldn't be connected
        edge_index = result.edge_index
        for i in range(edge_index.shape[1]):
            source = edge_index[0, i]
            dest = edge_index[1, i]
            # Same batch
            assert batch[source] == batch[dest]

    def test_forward_asserts_pos_tensor(self):
        """Test that forward raises AssertionError if pos is not Tensor."""
        data = Data()  # No pos attribute
        transform = BallGraph(r=1.0)

        with pytest.raises(AssertionError):
            transform(data)

    @pytest.mark.parametrize(
        "radius,expected_min_edges",
        [
            (0.5, 0),
            (1.5, 2),
            (3.0, 4),
        ],
    )
    def test_forward_different_radii(self, simple_data, radius, expected_min_edges):
        """Test forward pass with different radii."""
        transform = BallGraph(r=radius, max_num_neighbors=5)
        result = transform(simple_data)

        # Check that number of edges is as expected
        assert result.edge_index.shape[1] >= expected_min_edges
