import torch
from torch_geometric.data import Data

from clouds.transforms.cluster import ClusterSelect, _select_nth_node_per_cluster, _select_random_node_per_cluster


class TestSelectRandomNodePerCluster:
    """Tests for _select_random_node_per_cluster function."""

    def test_select_random_node_per_cluster_basic(self):
        """Test random selection from each cluster."""
        cluster = torch.tensor([0, 0, 1, 1, 2, 2, 2])

        # Since it's random, we can't test exact values, but we can test properties
        selection_index = _select_random_node_per_cluster(cluster)

        # Should return one index per cluster (3 clusters)
        assert selection_index.size(0) == 3

        # All indices should be valid (within range)
        assert torch.all(selection_index >= 0)
        assert torch.all(selection_index < cluster.size(0))

        # Each selected index should belong to the correct cluster
        selected_clusters = cluster[selection_index]
        expected_clusters = torch.unique(cluster)
        assert torch.all(selected_clusters.sort()[0] == expected_clusters.sort()[0])

        # Should have one selection per cluster
        assert len(torch.unique(selection_index)) == 3

    def test_select_random_node_per_cluster_single_cluster(self):
        """Test random selection when there's only one cluster."""
        cluster = torch.tensor([0, 0, 0, 0, 0])
        selection_index = _select_random_node_per_cluster(cluster)

        assert selection_index.size(0) == 1
        assert 0 <= selection_index[0] < cluster.size(0)
        assert cluster[selection_index[0]] == 0

    def test_select_random_node_per_cluster_many_clusters(self):
        """Test random selection with many clusters of size 1."""
        cluster = torch.arange(10)  # Each node is its own cluster
        selection_index = _select_random_node_per_cluster(cluster)

        assert selection_index.size(0) == 10
        assert torch.all(selection_index == torch.arange(10))

    def test_select_random_node_per_cluster_device(self):
        """Test that device is preserved."""
        cluster = torch.tensor([0, 0, 1, 1], device='cuda' if torch.cuda.is_available() else 'cpu')
        selection_index = _select_random_node_per_cluster(cluster)
        assert selection_index.device == cluster.device


class TestSelectNthNodePerCluster:
    """Tests for _select_nth_node_per_cluster function."""

    def test_select_nth_node_per_cluster_first(self):
        """Test selecting first node from each cluster (n=0)."""
        cluster = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        selection_index = _select_nth_node_per_cluster(cluster, 0)

        # Should select first node of each cluster: indices 0, 2, 5
        expected = torch.tensor([0, 2, 5])
        assert torch.all(selection_index == expected)

        # Verify clusters
        assert torch.all(cluster[selection_index] == torch.tensor([0, 1, 2]))

    def test_select_nth_node_per_cluster_second(self):
        """Test selecting second node from each cluster (n=1)."""
        cluster = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        selection_index = _select_nth_node_per_cluster(cluster, 1)

        # Should select second node of each cluster: indices 1, 3, 6
        expected = torch.tensor([1, 3, 6])
        assert torch.all(selection_index == expected)

        # Verify clusters
        assert torch.all(cluster[selection_index] == torch.tensor([0, 1, 2]))

    def test_select_nth_node_per_cluster_large_n(self):
        """Test with n larger than cluster size (should wrap around)."""
        cluster = torch.tensor([0, 0, 0, 1, 1])

        # n=3 should wrap to n=0 for cluster 0 (size 3), and n=1 for cluster 1 (size 2)
        selection_index = _select_nth_node_per_cluster(cluster, 3)

        # Cluster 0: size 3, 3%3=0 -> first node (index 0)
        # Cluster 1: size 2, 3%2=1 -> second node (index 4)
        expected = torch.tensor([0, 4])
        assert torch.all(selection_index == expected)

    def test_select_nth_node_per_cluster_large_clusters(self):
        """Test with larger clusters."""
        cluster = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])

        for n in range(4):
            selection_index = _select_nth_node_per_cluster(cluster, n)

            # Should select one from each cluster
            assert selection_index.size(0) == 3

            # All selected indices should be valid
            assert torch.all(selection_index >= 0)
            assert torch.all(selection_index < cluster.size(0))

            # Check that each cluster has exactly one selection
            selected_clusters = cluster[selection_index]
            unique_clusters, counts = torch.unique(selected_clusters, return_counts=True)
            assert torch.all(counts == 1)
            assert torch.all(unique_clusters == torch.tensor([0, 1, 2]))

    def test_select_nth_node_per_cluster_device(self):
        """Test that device is preserved."""
        cluster = torch.tensor([0, 0, 1, 1], device='cuda' if torch.cuda.is_available() else 'cpu')
        selection_index = _select_nth_node_per_cluster(cluster, 0)
        assert selection_index.device == cluster.device


class TestClusterSelect:
    """Tests for ClusterSelect transform."""

    def test_cluster_select_random(self):
        """Test random selection mode."""
        data = Data(
            x=torch.randn(10, 5),
            cluster=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
            batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        )

        transform = ClusterSelect(deterministic=False)
        result = transform(data)

        # Should have selection_index
        assert hasattr(result, 'selection_index')

        # Should select one per cluster
        assert result.selection_index.size(0) == 4  # 4 clusters

        # All selected indices should be valid
        assert torch.all(result.selection_index >= 0)
        assert torch.all(result.selection_index < 10)

        # Each cluster should have exactly one selection
        selected_clusters = result.cluster[result.selection_index]
        unique_clusters, counts = torch.unique(selected_clusters, return_counts=True)
        assert torch.all(counts == 1)
        assert torch.all(unique_clusters == torch.tensor([0, 1, 2, 3]))

    def test_cluster_select_deterministic_with_pick(self):
        """Test deterministic selection with fixed pick."""
        data = Data(
            x=torch.randn(10, 5),
            cluster=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
            batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        )

        transform = ClusterSelect(deterministic=True, pick=1)
        result = transform(data)

        # Should select second node from each cluster (n=1)
        # Cluster 0 (indices 0-1): select 1
        # Cluster 1 (indices 2-4): select 3
        # Cluster 2 (indices 5-8): select 6
        # Cluster 3 (index 9): select 9
        expected = torch.tensor([1, 3, 6, 9])
        assert torch.all(result.selection_index == expected)

    def test_cluster_select_deterministic_without_pick(self):
        """Test deterministic selection with auto-incrementing pick."""
        data = Data(
            x=torch.randn(10, 5),
            cluster=torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3]),
            batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        )

        transform = ClusterSelect(deterministic=True)

        # First call should use pick=0
        result1 = transform(data)
        expected1 = torch.tensor([0, 2, 5, 9])
        assert torch.all(result1.selection_index == expected1)

        # Second call should use pick=1 (auto-incremented)
        result2 = transform(data)
        expected2 = torch.tensor([1, 3, 6, 9])
        assert torch.all(result2.selection_index == expected2)

        # Third call should use pick=2
        result3 = transform(data)
        expected3 = torch.tensor([0, 4, 7, 9])
        assert torch.all(result3.selection_index == expected3)

    def test_cluster_select_single_cluster(self):
        """Test with a single cluster."""
        data = Data(x=torch.randn(5, 3), cluster=torch.tensor([0, 0, 0, 0, 0]), batch=torch.tensor([0, 0, 0, 0, 0]))

        transform = ClusterSelect(deterministic=True, pick=2)
        result = transform(data)

        # Should select one node from the single cluster
        assert result.selection_index.size(0) == 1
        assert result.selection_index[0] == 2  # pick=2 selects index 2 (0-indexed)

        # Random selection should also work
        transform_random = ClusterSelect(deterministic=False)
        result_random = transform_random(data)
        assert result_random.selection_index.size(0) == 1
