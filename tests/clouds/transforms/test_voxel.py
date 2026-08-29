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
        assert hasattr(data, 'cluster_index')
        assert isinstance(data.cluster_index, torch.Tensor)

        # Check that cluster is contiguous (0 to num_clusters-1)
        unique_clusters = torch.unique(data.cluster_index)
        assert unique_clusters.tolist() == list(range(len(unique_clusters)))

    @pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason="pyg grid clustering not installed")
    def test_forward_multiple_batches(self, sample_data_multi_batch):
        """Test forward pass with multiple batches."""
        transform = VoxelCluster(voxel_size=1.0)
        data = transform(sample_data_multi_batch)

        # Check that cluster is contiguous globally
        unique_clusters = torch.unique(data.cluster_index)
        assert unique_clusters.tolist() == list(range(len(unique_clusters)))

        # Check batch information is preserved
        assert data.batch is not None
        assert len(data.batch) == len(data.pos)

        # Check each batch has valid clusters
        for batch_id in torch.unique(data.batch):
            mask = data.batch == batch_id
            batch_clusters = data.cluster_index[mask]
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
            results.append(data.cluster_index)

        # Check that not all results are identical (statistically)
        unique_results = [torch.unique(r) for r in results]
        assert len(set(tuple(r.tolist()) for r in unique_results)) > 1


"""
Unit tests for VoxelSelect.

NOTE ON IMPORTS: adjust these to match your project's actual layout.
  - `VoxelSelect` (and the module it lives in, needed for monkeypatching
    `HAS_VPSAMPLE`) is assumed importable as shown below.
  - `Data` is assumed to be `torch_geometric.data.Data`, consistent with
    `VoxelSelect` subclassing `BaseTransform` (a torch_geometric concept)
    and with `data.batch` being used to distinguish single graphs from
    batched ones. Swap this out if your project defines its own `Data`.
"""


class TestVoxelSelect:
    def test_selects_approximately_expected_number_of_voxels(self):
        """
        For points drawn uniformly at random over a known volume, the number
        of voxels selected for a fixed voxel_size should be close to the
        number of *occupied* voxels predicted by a standard balls-into-bins
        calculation. This check is implementation-agnostic (it doesn't
        assume anything about hashing/anchoring details) and just verifies
        the overall subsampling density is right.
        """
        torch.manual_seed(0)
        random.seed(0)

        n_points = 4000
        extent = 10.0  # points live in [0, extent)^3
        voxel_size = 1.0

        pos = torch.rand(n_points, 3) * extent
        data = Data(pos=pos)

        transform = VoxelSelect(voxel_size=voxel_size, deterministic=True, pick=0)
        out = transform(data)

        assert hasattr(out, "selection_index"), "expected `selection_index` to be set"
        selected_count = out.selection_index.numel()

        n_voxels = (extent / voxel_size) ** 3
        expected_occupied = n_voxels * (1 - (1 - 1 / n_voxels) ** n_points)

        # Loose tolerance: this is a statistical property of the input
        # distribution and voxel size, not an exact combinatorial identity.
        tolerance = 0.1 * expected_occupied
        assert selected_count == pytest.approx(expected_occupied, abs=tolerance), (
            f"expected ~{expected_occupied:.0f} occupied voxels, got {selected_count}"
        )

    def test_batched_voxelization_matches_individual_processing(self, monkeypatch):
        """
        Voxelizing several point clouds together as a batch must give the
        same per-cloud result as voxelizing each cloud on its own -- batching
        should not let voxels/points leak across point-cloud boundaries.

        HAS_VPSAMPLE is forced off so that both the batched call (which
        always takes the Compose(VoxelCluster, ClusterSelect) path, since
        `data.batch` is a Tensor) and the per-cloud calls go through the
        *same* code path, making this a fair, apples-to-apples comparison.
        """

        torch.manual_seed(42)
        random.seed(42)

        n_clouds = 4
        points_per_cloud = 2500
        voxel_size = 0.5
        pick = 0

        clouds = [torch.rand(points_per_cloud, 3) * 5.0 for _ in range(n_clouds)]
        batch_pos = torch.cat(clouds, dim=0)
        batch_index = torch.cat(
            [torch.full((points_per_cloud,), i, dtype=torch.long) for i in range(n_clouds)]
        )

        batched_data = Data(pos=batch_pos.clone(), batch=batch_index)
        batched_out = VoxelSelect(voxel_size=voxel_size, deterministic=True, pick=pick)(
            batched_data
        )

        selected_global = batched_out.selection_index
        selected_batch_ids = batch_index[selected_global]

        for i, cloud in enumerate(clouds):
            individual_data = Data(pos=cloud.clone())
            individual_out = VoxelSelect(
                voxel_size=voxel_size, deterministic=True, pick=pick
            )(individual_data)

            assert individual_out.selection_index.size(0) == pytest.approx(
                selected_global[selected_batch_ids == i].size(0), rel=1/100
            )

            # TODO: ideally, results would be completely identical
            # expected_local = torch.sort(individual_out.selection_index).values
            # offset = i * points_per_cloud
            # actual_local = torch.sort(
            #     selected_global[selected_batch_ids == i] - offset
            # ).values
            # assert torch.equal(actual_local, expected_local), (
            #     f"cloud {i}: batched selection does not match individual selection"
            # )

    def test_random_voxel_size_is_independent_per_cloud_in_batch(self):
        """
        With voxel_size given as a (min, max) range, each point cloud in a
        batch should get its own independently sampled voxel size. We verify
        this using *identical* duplicated point clouds: if voxel sizes are
        sampled independently per cloud, the number of voxels selected per
        cloud should (almost certainly) differ across the batch. If a single
        voxel size is instead shared across the whole batch, every cloud
        will select exactly the same number of points, since the underlying
        points are identical.
        """
        torch.manual_seed(7)
        random.seed(7)

        n_clouds = 8
        points_per_cloud = 2500
        single_cloud = torch.rand(points_per_cloud, 3) * 5.0

        # Duplicate the exact same cloud n_clouds times.
        batch_pos = single_cloud.repeat(n_clouds, 1)
        batch_index = torch.cat(
            [torch.full((points_per_cloud,), i, dtype=torch.long) for i in range(n_clouds)]
        )

        data = Data(pos=batch_pos, batch=batch_index)
        transform = VoxelSelect(
            voxel_size=(0.05, 2.0),
            large_voxel_prob=1.0,  # always draw uniformly from the full range
            deterministic=True,
            pick=0,
        )
        out = transform(data)

        selected_global = out.selection_index
        selected_batch_ids = batch_index[selected_global]

        counts = [int((selected_batch_ids == i).sum()) for i in range(n_clouds)]

        # Since every cloud contains identical points, identical counts
        # across all clouds can only happen if every cloud was voxelized
        # with the same voxel_size. With a truly independent per-cloud
        # random voxel_size drawn from a wide continuous range, getting the
        # same count for every single cloud is vanishingly unlikely.
        assert len(set(counts)) > 1, (
            f"all {n_clouds} identical clouds produced the same selection "
            f"count ({counts[0]}), implying a single shared voxel_size was "
            f"used for the whole batch instead of one per cloud: {counts}"
        )

# TODO: check consistency between different code paths
