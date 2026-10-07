import random

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.typing import WITH_GRID_CLUSTER as HAS_PYG_GRID_CLUSTER

from clouds.transforms.voxel import VoxelCluster, VoxelSelect


@pytest.fixture
def sample_data():
    return Data(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [2.0, 2.0, 2.0],
            ]
        )
    )


@pytest.fixture
def sample_data_multi_batch():
    return Batch(
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [2.0, 2.0, 2.0],
                [2.1, 2.1, 2.1],
            ]
        ),
        batch=torch.tensor([0, 0, 0, 1, 1, 1]),
    )


@pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason='pyg grid clustering not installed')
def test_voxel_cluster_forward_single_batch(sample_data):
    data = VoxelCluster(voxel_size=1.0)(sample_data)

    unique_clusters = torch.unique(data.cluster_index)
    assert unique_clusters.tolist() == list(range(len(unique_clusters)))


@pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason='pyg grid clustering not installed')
def test_voxel_cluster_forward_multiple_batches(sample_data_multi_batch):
    data = VoxelCluster(voxel_size=1.0)(sample_data_multi_batch)

    unique_clusters = torch.unique(data.cluster_index)
    assert unique_clusters.tolist() == list(range(len(unique_clusters)))
    assert len(data.batch) == len(data.pos)
    for batch_id in torch.unique(data.batch):
        assert len(torch.unique(data.cluster_index[data.batch == batch_id])) > 0


@pytest.mark.skipif(not HAS_PYG_GRID_CLUSTER, reason='pyg grid clustering not installed')
def test_voxel_cluster_random_voxel_size_tuple(sample_data):
    random.seed(42)
    transform = VoxelCluster(voxel_size=(0.1, 0.5))

    unique_results = [torch.unique(transform(sample_data.clone()).cluster_index) for _ in range(5)]

    assert len({tuple(result.tolist()) for result in unique_results}) > 1


def test_voxel_select_selects_approximately_expected_number_of_voxels():
    n_points, extent, voxel_size = 4000, 10.0, 1.0

    selected_count = VoxelSelect(voxel_size=voxel_size, deterministic=True, pick=0)(
        Data(pos=torch.rand(n_points, 3) * extent)
    ).selection_index.numel()

    n_voxels = (extent / voxel_size) ** 3
    expected_occupied = n_voxels * (1 - (1 - 1 / n_voxels) ** n_points)

    assert selected_count == pytest.approx(expected_occupied, abs=0.1 * expected_occupied)


def test_voxel_select_batched_voxelization_matches_individual_processing():
    n_clouds, points_per_cloud, voxel_size = 4, 2500, 0.5

    clouds = [torch.rand(points_per_cloud, 3) * 5.0 for _ in range(n_clouds)]
    batch_index = torch.cat([torch.full((points_per_cloud,), i, dtype=torch.long) for i in range(n_clouds)])

    batched_out = VoxelSelect(voxel_size=voxel_size, deterministic=True, pick=0)(Data(pos=torch.cat(clouds), batch=batch_index))
    selected_global = batched_out.selection_index
    selected_batch_ids = batch_index[selected_global]

    for i, cloud in enumerate(clouds):
        individual_count = VoxelSelect(voxel_size=voxel_size, deterministic=True, pick=0)(
            Data(pos=cloud.clone())
        ).selection_index.size(0)
        assert individual_count == pytest.approx(selected_global[selected_batch_ids == i].size(0), rel=1 / 100)


def test_voxel_select_random_voxel_size_is_independent_per_cloud_in_batch():
    n_clouds, points_per_cloud = 8, 250000
    single_cloud = torch.rand(points_per_cloud, 3) * 5.0
    batch_index = torch.cat([torch.full((points_per_cloud,), i, dtype=torch.long) for i in range(n_clouds)])

    out = VoxelSelect(voxel_size=(0.05, 2.0), large_voxel_prob=1.0, deterministic=True, pick=0)(
        Data(pos=single_cloud.repeat(n_clouds, 1), batch=batch_index)
    )

    selected_batch_ids = batch_index[out.selection_index]

    # Identical clouds can only produce identical counts if a single voxel_size was shared across the batch.
    assert len({int((selected_batch_ids == i).sum()) for i in range(n_clouds)}) > 1
