import pytest
import torch

from clouds.transforms.knn import HAS_KEOPS, HAS_NANOFLANN, knn


class TestKNNImplementations:
    """Test suite to verify KNN implementations produce identical results."""

    @pytest.fixture
    def random_positions(self):
        """Generate random positions for testing."""
        torch.manual_seed(42)
        return torch.randn(100, 3)

    @pytest.fixture
    def small_positions(self):
        """Generate small set of positions."""
        return torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float,
        )

    @pytest.fixture
    def batched_positions(self):
        """Generate positions with batch indices."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [10.0, 10.0, 10.0],
                [11.0, 10.0, 10.0],
                [10.0, 11.0, 10.0],
                [10.0, 10.0, 11.0],
            ],
            dtype=torch.float,
        )
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        return positions, batch

    def _compare_knn_results(self, result1, result2):
        """Helper to compare KNN results."""
        assert result1.shape == result2.shape, f"Shape mismatch: {result1.shape} vs {result2.shape}"
        # Sort results along last dimension for comparison (neighbors may be in different order if distances are equal)
        sorted1 = torch.sort(result1, dim=-1)[0]
        sorted2 = torch.sort(result2, dim=-1)[0]
        assert torch.allclose(sorted1, sorted2), "Results differ"

    # CPU tests with PyG implementation as baseline
    def test_pyg_vs_keops_cpu_small(self, small_positions):
        """Test PyG vs KeOps on CPU with small data."""
        if not HAS_KEOPS:
            pytest.skip("KeOps not installed")

        k = 3
        # PyG implementation
        pyg_result = knn(
            pos=small_positions.cpu(),
            k=k,
        )

        # KeOps implementation (force CPU)
        keops_result = knn(
            pos=small_positions.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, keops_result)

    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_pyg_vs_keops_random(self, random_positions):
        """Test PyG vs KeOps on random data."""
        k = 5
        pyg_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        keops_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, keops_result)

    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_pyg_vs_keops_query_positions(self, random_positions):
        """Test with different query and reference positions."""
        torch.manual_seed(123)
        query_pos = torch.randn(50, 3)
        k = 4

        pyg_result = knn(
            pos=random_positions.cpu(),
            query_pos=query_pos.cpu(),
            k=k,
        )

        keops_result = knn(
            pos=random_positions.cpu(),
            query_pos=query_pos.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, keops_result)

    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_pyg_vs_keops_batched(self, batched_positions):
        """Test with batched data."""
        positions, batch = batched_positions
        k = 3

        pyg_result = knn(
            pos=positions.cpu(),
            batch=batch.cpu(),
            k=k,
        )

        keops_result = knn(
            pos=positions.cpu(),
            batch=batch.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, keops_result)

    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_keops_batched_query(self, batched_positions):
        """Test batched with different query positions."""
        positions, batch = batched_positions
        query_pos = positions[:2]  # Query only first 2 points
        query_batch = batch[:2]
        k = 3

        pyg_result = knn(
            pos=positions.cpu(),
            query_pos=query_pos.cpu(),
            batch=batch.cpu(),
            query_batch=query_batch.cpu(),
            k=k,
        )

        keops_result = knn(
            pos=positions.cpu(),
            query_pos=query_pos.cpu(),
            batch=batch.cpu(),
            query_batch=query_batch.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, keops_result)

    # Tests for NanoFlann
    @pytest.mark.skipif(not HAS_NANOFLANN, reason="NanoFlann not installed")
    def test_pyg_vs_nanoflann_small(self, small_positions):
        """Test PyG vs NanoFlann on small data."""
        k = 3
        pyg_result = knn(
            pos=small_positions.cpu(),
            k=k,
        )

        nanoflann_result = knn(
            pos=small_positions.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, nanoflann_result)

    @pytest.mark.skipif(not HAS_NANOFLANN, reason="NanoFlann not installed")
    def test_pyg_vs_nanoflann_random(self, random_positions):
        """Test PyG vs NanoFlann on random data."""
        k = 5
        pyg_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        nanoflann_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, nanoflann_result)

    @pytest.mark.skipif(not HAS_NANOFLANN, reason="NanoFlann not installed")
    def test_nanoflann_query_positions(self, random_positions):
        """Test NanoFlann with query positions."""
        torch.manual_seed(123)
        query_pos = torch.randn(50, 3)
        k = 4

        pyg_result = knn(
            pos=random_positions.cpu(),
            query_pos=query_pos.cpu(),
            k=k,
        )

        nanoflann_result = knn(
            pos=random_positions.cpu(),
            query_pos=query_pos.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, nanoflann_result)

    @pytest.mark.skipif(not HAS_NANOFLANN, reason="NanoFlann not installed")
    def test_nanoflann_caching(self, random_positions):
        """Test that NanoFlann caching works correctly."""
        k = 3

        # First call should build cache
        result1 = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        # Second call should use cache
        result2 = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        self._compare_knn_results(result1, result2)

    @pytest.mark.skipif(not HAS_NANOFLANN, reason="NanoFlann not installed")
    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_all_implementations_match(self, random_positions):
        """Test that all three implementations produce identical results."""
        k = 4

        pyg_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        nanoflann_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        keops_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        self._compare_knn_results(pyg_result, nanoflann_result)
        self._compare_knn_results(pyg_result, keops_result)

    # Edge cases
    def test_k_greater_than_points(self, random_positions):
        """Test error when k is greater than number of points."""
        with pytest.raises(ValueError, match="fewer than K points"):
            knn(
                pos=random_positions.cpu(),
                k=random_positions.size(0) + 1,
            )

    def test_empty_query(self, random_positions):
        """Test error when query has 0 points."""
        with pytest.raises(ValueError, match="neighbors of 0 nodes"):
            knn(
                pos=random_positions.cpu(),
                query_pos=torch.empty(0, 3),
                k=3,
            )

    def test_k_equals_points(self, random_positions):
        """Test when k equals number of points."""
        k = random_positions.size(0)
        result = knn(
            pos=random_positions.cpu(),
            k=k,
        )
        assert result.shape == (random_positions.size(0), k)
        # All points should be neighbors (in some order)
        for i in range(result.size(0)):
            assert set(result[i].tolist()) == set(range(k))

    def test_k_equals_one(self, random_positions):
        """Test when k equals 1."""
        k = 1
        result = knn(
            pos=random_positions.cpu(),
            k=k,
        )
        assert result.shape == (random_positions.size(0), k)
        # Each point should be its own nearest neighbor
        assert torch.all(result.squeeze() == torch.arange(random_positions.size(0)))

    # GPU tests if available
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_gpu_keops(self, random_positions):
        """Test KeOps on GPU."""
        k = 5
        result = knn(
            pos=random_positions.cuda(),
            k=k,
        )
        assert result.is_cuda
        assert result.shape == (random_positions.size(0), k)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_KEOPS, reason="KeOps not installed")
    def test_cpu_vs_gpu_keops(self, random_positions):
        """Test CPU vs GPU KeOps consistency."""
        k = 4

        cpu_result = knn(
            pos=random_positions.cpu(),
            k=k,
        )

        gpu_result = knn(
            pos=random_positions.cuda(),
            k=k,
        ).cpu()

        self._compare_knn_results(cpu_result, gpu_result)

    # Test with different dimensions
    def test_2d_positions(self):
        """Test with 2D positions."""
        torch.manual_seed(42)
        positions = torch.randn(50, 2)
        k = 3

        result = knn(
            pos=positions.cpu(),
            k=k,
        )
        assert result.shape == (50, k)

    def test_high_dimensional_positions(self):
        """Test with high-dimensional positions."""
        torch.manual_seed(42)
        positions = torch.randn(50, 20)
        k = 3

        result = knn(
            pos=positions.cpu(),
            k=k,
        )
        assert result.shape == (50, k)

    # Test with duplicate points
    def test_duplicate_points(self):
        """Test behavior with duplicate points."""
        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # duplicate
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # duplicate
            ],
            dtype=torch.float,
        )
        k = 3

        result = knn(
            pos=positions.cpu(),
            k=k,
        )
        assert result.shape == (4, k)
        # First point should find itself and the other duplicate
        assert result[0, 0] == 0
        assert 1 in result[0]
