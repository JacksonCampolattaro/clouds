import pytest
import torch
from torch_geometric.data import Data

from clouds.transforms.cluster import ClusterSelect, _select_nth_node_per_cluster, _select_random_node_per_cluster


def test_select_random_node_per_cluster_basic():
    cluster = torch.tensor([0, 0, 1, 1, 2, 2, 2])

    selection_index = _select_random_node_per_cluster(cluster)

    assert selection_index.size(0) == 3
    assert torch.all(selection_index >= 0)
    assert torch.all(selection_index < cluster.size(0))
    assert torch.equal(cluster[selection_index].sort()[0], torch.unique(cluster).sort()[0])
    assert len(torch.unique(selection_index)) == 3


def test_select_random_node_per_cluster_single_cluster():
    cluster = torch.tensor([0, 0, 0, 0, 0])

    selection_index = _select_random_node_per_cluster(cluster)

    assert selection_index.size(0) == 1
    assert 0 <= selection_index[0] < 5
    assert cluster[selection_index[0]] == 0


def test_select_random_node_per_cluster_many_clusters():
    selection_index = _select_random_node_per_cluster(torch.arange(10))

    assert torch.equal(selection_index, torch.arange(10))


def test_select_random_node_per_cluster_device():
    cluster = torch.tensor([0, 0, 1, 1], device='cuda' if torch.cuda.is_available() else 'cpu')

    assert _select_random_node_per_cluster(cluster).device == cluster.device


def test_select_nth_node_per_cluster_first():
    cluster = torch.tensor([0, 0, 1, 1, 1, 2, 2])

    assert torch.equal(_select_nth_node_per_cluster(cluster, 0), torch.tensor([0, 2, 5]))


def test_select_nth_node_per_cluster_second():
    cluster = torch.tensor([0, 0, 1, 1, 1, 2, 2])

    assert torch.equal(_select_nth_node_per_cluster(cluster, 1), torch.tensor([1, 3, 6]))


def test_select_nth_node_per_cluster_large_n():
    # Cluster 0 has size 3 (3 % 3 = 0), cluster 1 has size 2 (3 % 2 = 1).
    cluster = torch.tensor([0, 0, 0, 1, 1])

    assert torch.equal(_select_nth_node_per_cluster(cluster, 3), torch.tensor([0, 4]))


def test_select_nth_node_per_cluster_large_clusters():
    cluster = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])

    for n in range(4):
        selection_index = _select_nth_node_per_cluster(cluster, n)

        assert selection_index.size(0) == 3
        assert torch.all(selection_index >= 0)
        assert torch.all(selection_index < cluster.size(0))
        unique_clusters, counts = torch.unique(cluster[selection_index], return_counts=True)
        assert torch.all(counts == 1)
        assert torch.equal(unique_clusters, torch.tensor([0, 1, 2]))


def test_select_nth_node_per_cluster_device():
    cluster = torch.tensor([0, 0, 1, 1], device='cuda' if torch.cuda.is_available() else 'cpu')

    assert _select_nth_node_per_cluster(cluster, 0).device == cluster.device


@pytest.fixture
def data():
    return Data(
        x=torch.randn(10, 5),
        cluster_index=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    )


@pytest.fixture
def non_contiguous_data():
    return Data(
        x=torch.randn(15, 5),
        cluster_index=torch.tensor([0, 0, 3, 3, 3, 1, 1, 2, 2, 2, 2, 4, 5, 4, 5]),
        batch=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]),
    )


def test_cluster_select_random(data):
    result = ClusterSelect(deterministic=False)(data)

    assert result.selection_index.size(0) == 4
    assert torch.all(result.selection_index >= 0)
    assert torch.all(result.selection_index < 10)
    unique_clusters, counts = torch.unique(result.cluster_index[result.selection_index], return_counts=True)
    assert torch.all(counts == 1)
    assert torch.equal(unique_clusters, torch.tensor([0, 1, 2, 3]))


def test_cluster_select_deterministic_with_pick(data):
    assert torch.equal(ClusterSelect(deterministic=True, pick=1)(data).selection_index, torch.tensor([1, 3, 6, 9]))


def test_cluster_select_deterministic_without_pick(data):
    transform = ClusterSelect(deterministic=True)

    assert torch.equal(transform(data).selection_index, torch.tensor([0, 2, 5, 9]))
    assert torch.equal(transform(data).selection_index, torch.tensor([1, 3, 6, 9]))
    assert torch.equal(transform(data).selection_index, torch.tensor([0, 4, 7, 9]))


def test_cluster_select_single_cluster():
    data = Data(x=torch.randn(5, 3), cluster_index=torch.tensor([0, 0, 0, 0, 0]), batch=torch.tensor([0, 0, 0, 0, 0]))

    assert ClusterSelect(deterministic=True, pick=2)(data).selection_index.tolist() == [2]
    assert ClusterSelect(deterministic=False)(data).selection_index.size(0) == 1


def test_cluster_select_deterministic_non_contiguous_with_batch(non_contiguous_data):
    result = ClusterSelect(deterministic=True, pick=1)(non_contiguous_data)

    assert result.selection_index.size(0) == 6
    unique_clusters, counts = torch.unique(result.cluster_index[result.selection_index], return_counts=True)
    assert torch.all(counts == 1)
    assert torch.equal(unique_clusters, torch.tensor([0, 1, 2, 3, 4, 5]))


def test_cluster_select_random_non_contiguous_with_batch(non_contiguous_data):
    result = ClusterSelect(deterministic=False)(non_contiguous_data)

    assert result.selection_index.size(0) == 6
    unique_clusters, counts = torch.unique(result.cluster_index[result.selection_index], return_counts=True)
    assert torch.all(counts == 1)
    assert torch.equal(unique_clusters, torch.tensor([0, 1, 2, 3, 4, 5]))
