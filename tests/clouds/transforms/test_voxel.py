import random

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.typing import WITH_GRID_CLUSTER as HAS_PYG_GRID_CLUSTER

from clouds.transforms.voxel import VoxelCluster, VoxelSelect


@pytest.fixture
def sample_data():
    """Create sample point cloud data."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [2.0, 2.0, 2.0],
        ],
        dtype=torch.float,
    )
    return Data(pos=pos)


@pytest.fixture
def sample_data_multi_batch():
    """Create sample point cloud data with multiple batches."""
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [2.0, 2.0, 2.0],
            [2.1, 2.1, 2.1],
        ],
        dtype=torch.float,
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    return Batch(pos=pos, batch=batch)


@pytest.fixture
def sample_data_gpu(sample_data):
    """Create sample data on GPU if available."""
    if torch.cuda.is_available():
        sample_data.pos = sample_data.pos.cuda()
        sample_data.batch = torch.zeros(len(sample_data.pos), dtype=torch.long).cuda()
    return sample_data


class TestVoxelCluster:
    """Tests for VoxelCluster class."""

    @pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason="pyg grid clustering not installed")
    def test_forward_single_batch(self, sample_data):
        """Test forward pass with single batch."""
        transform = VoxelCluster(voxel_size=1.0)
        data = transform(sample_data)

        # Check that cluster is added
        assert hasattr(data, 'cluster')
        assert isinstance(data.cluster, torch.Tensor)

        # Check that cluster is contiguous (0 to num_clusters-1)
        unique_clusters = torch.unique(data.cluster)
        assert unique_clusters.tolist() == list(range(len(unique_clusters)))

        # Check that data is sorted by cluster
        assert torch.all(data.cluster[:-1] <= data.cluster[1:])

    @pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason="pyg grid clustering not installed")
    def test_forward_multiple_batches(self, sample_data_multi_batch):
        """Test forward pass with multiple batches."""
        transform = VoxelCluster(voxel_size=1.0)
        data = transform(sample_data_multi_batch)

        # Check that cluster is contiguous globally
        unique_clusters = torch.unique(data.cluster)
        assert unique_clusters.tolist() == list(range(len(unique_clusters)))

        # Check batch information is preserved
        assert data.batch is not None
        assert len(data.batch) == len(data.pos)

        # Check each batch has valid clusters
        for batch_id in torch.unique(data.batch):
            mask = data.batch == batch_id
            batch_clusters = data.cluster[mask]
            assert len(torch.unique(batch_clusters)) > 0

    @pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason="pyg grid clustering not installed")
    def test_random_voxel_size_tuple(self, sample_data):
        """Test that tuple voxel size produces random values."""
        transform = VoxelCluster(voxel_size=(0.1, 0.5))

        # Run multiple times and check different results
        results = []
        random.seed(42)  # Set seed for reproducibility
        for _ in range(5):
            data = transform(sample_data.clone())
            results.append(data.cluster)

        # Check that not all results are identical (statistically)
        unique_results = [torch.unique(r) for r in results]
        assert len(set(tuple(r.tolist()) for r in unique_results)) > 1


class TestVoxelSelect:
    """Tests for VoxelSelect class."""

    def test_pick_increment(self, sample_data):
        """Test that current_pick increments correctly."""
        transform = VoxelSelect(voxel_size=1.0)

        # First call
        transform(sample_data.clone())
        assert transform.current_pick == 1

        # Second call
        transform(sample_data.clone())
        assert transform.current_pick == 2


# TODO: check consistency between different code paths
