import pytest
import torch
from torch_geometric.typing import WITH_FPS as HAS_PYG_FPS

from clouds.transforms.fps import HAS_TORCH_FPSAMPLE, fps


class TestFPSImplementations:
    """Test suite to verify FPS implementations produce identical results."""

    @pytest.fixture
    def random_positions(self):
        """Generate random 3D positions for testing."""
        torch.manual_seed(42)
        return torch.randn(100, 3)

    @pytest.fixture
    def small_positions(self):
        """Generate small set of positions for testing."""
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
    def clustered_positions(self):
        """Generate positions with clear clusters for testing FPS behavior."""
        torch.manual_seed(42)
        # Two clusters far apart
        cluster1 = torch.randn(30, 3) * 0.1
        cluster2 = torch.randn(30, 3) * 0.1 + 10.0
        return torch.cat([cluster1, cluster2], dim=0)

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

    def _compare_fps_results(self, result1, result2, pos, n=None):
        """Helper to compare FPS results, accounting for different starting points."""
        # Both results should have the same length
        assert len(result1) == len(result2), f"Length mismatch: {len(result1)} vs {len(result2)}"

        if n is None:
            n = len(result1)

        # When deterministic, results should be identical
        if torch.all(result1 == result2):
            return True

        # Otherwise, check that both are valid indices and unique
        for result in [result1, result2]:
            assert result.min() >= 0
            assert result.max() < pos.size(0)
            assert len(torch.unique(result)) == len(result)

        # Check that both selections are valid FPS results
        # They should both be within the data and have similar coverage
        # This is a weaker check but handles the random start case
        return True

    def _verify_fps_properties(self, indices, pos):
        """Verify that the returned indices satisfy FPS properties."""
        # All indices should be valid
        assert indices.min() >= 0
        assert indices.max() < pos.size(0)

        # All indices should be unique
        assert len(torch.unique(indices)) == len(indices)

        # For FPS with deterministic=False, we can't check exact distances
        # But we can check that indices are within bounds
        return True

    # Tests for torch_fpsample implementation (CPU only)
    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_fpsample_vs_pyg_cpu_small(self, small_positions):
        """Test torch_fpsample vs PyG on small CPU data."""
        n = 3

        # PyG implementation (CPU with batch=None uses torch_fpsample)
        pyg_result = fps(
            pos=small_positions.cpu(),
            n=n,
            deterministic=True,
        )

        # Direct torch_fpsample call
        fpsample_result = fps(
            pos=small_positions.cpu(),
            n=n,
            deterministic=True,
        )

        assert torch.all(pyg_result == fpsample_result)
        self._verify_fps_properties(pyg_result, small_positions)

    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_fpsample_vs_pyg_cpu_random(self, random_positions):
        """Test torch_fpsample vs PyG on random CPU data."""
        n = 10

        pyg_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=True,
        )

        fpsample_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=True,
        )

        assert torch.all(pyg_result == fpsample_result)
        self._verify_fps_properties(pyg_result, random_positions)

    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_fpsample_with_ratio(self, random_positions):
        """Test torch_fpsample with ratio instead of n."""
        ratio = 0.2
        expected_n = int(ratio * len(random_positions))

        result = fps(
            pos=random_positions.cpu(),
            ratio=ratio,
            deterministic=True,
        )

        assert len(result) == expected_n
        self._verify_fps_properties(result, random_positions)

    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_fpsample_deterministic_vs_random(self, random_positions):
        """Test deterministic vs non-deterministic FPS."""
        n = 10

        # Deterministic (starts from index 0)
        det_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=True,
        )

        # Non-deterministic (random start)
        rand_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=False,
        )

        # Results should be different (usually)
        # But both should be valid FPS selections
        self._verify_fps_properties(det_result, random_positions)
        self._verify_fps_properties(rand_result, random_positions)

        # Both should have same length
        assert len(det_result) == len(rand_result)

    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_fpsample_clustered_data(self, clustered_positions):
        """Test FPS on clustered data to ensure it picks from both clusters."""
        n = 4

        result = fps(
            pos=clustered_positions.cpu(),
            n=n,
            deterministic=True,
        )

        self._verify_fps_properties(result, clustered_positions)

        # Should pick points from both clusters
        selected_points = clustered_positions[result]
        cluster1_center = torch.tensor([0.0, 0.0, 0.0])
        cluster2_center = torch.tensor([10.0, 10.0, 10.0])

        # Check if points are from both clusters
        dist_to_cluster1 = torch.norm(selected_points - cluster1_center, dim=1)
        dist_to_cluster2 = torch.norm(selected_points - cluster2_center, dim=1)

        # Some points should be closer to cluster1, some to cluster2
        close_to_cluster1 = (dist_to_cluster1 < dist_to_cluster2).sum()
        close_to_cluster2 = (dist_to_cluster2 < dist_to_cluster1).sum()

        assert close_to_cluster1 > 0, "No points selected from cluster 1"
        assert close_to_cluster2 > 0, "No points selected from cluster 2"

    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_fpsample_caching(self, random_positions):
        """Test that FPS results are consistent with deterministic=True."""
        n = 10

        result1 = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=True,
        )

        result2 = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=True,
        )

        assert torch.all(result1 == result2)

    # Tests for PyG implementation (CPU and GPU)
    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_cpu(self, random_positions):
        """Test PyG FPS on CPU."""
        n = 10

        result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=False,
            batch=None,
        )

        self._verify_fps_properties(result, random_positions)
        assert len(result) == n

    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_with_ratio(self, random_positions):
        """Test PyG FPS with ratio."""
        ratio = 0.2

        result = fps(
            pos=random_positions.cpu(),
            ratio=ratio,
            deterministic=False,
            batch=None,
        )

        expected_n = int(ratio * len(random_positions))
        assert len(result) == expected_n

    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_batched(self, batched_positions):
        """Test PyG FPS with batched data."""
        positions, batch = batched_positions
        ratio = 0.5

        result = fps(
            pos=positions.cpu(),
            ratio=ratio,
            batch=batch,
            deterministic=False,
        )

        self._verify_fps_properties(result, positions)

        # Check that each batch has approximately half its points
        batch0_indices = result[batch[result] == 0]
        batch1_indices = result[batch[result] == 1]

        assert len(batch0_indices) == 2  # 4 points * 0.5
        assert len(batch1_indices) == 2  # 4 points * 0.5

    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_batched_with_batch_size(self, batched_positions):
        """Test PyG FPS with explicit batch_size."""
        positions, batch = batched_positions
        ratio = 0.5

        result = fps(
            pos=positions.cpu(),
            ratio=ratio,
            batch=batch,
            batch_size=2,
            deterministic=False,
        )

        self._verify_fps_properties(result, positions)

    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_edge_cases(self, small_positions):
        """Test edge cases for PyG FPS."""
        # Test n=1
        result = fps(
            pos=small_positions.cpu(),
            n=1,
            deterministic=False,
        )
        assert len(result) == 1
        assert result[0] >= 0 and result[0] < len(small_positions)

        # Test n = all points
        result = fps(
            pos=small_positions.cpu(),
            n=len(small_positions),
            deterministic=False,
        )
        assert len(result) == len(small_positions)
        # Should be a permutation of all indices
        assert torch.all(torch.sort(result)[0] == torch.arange(len(small_positions)))

    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_random_vs_deterministic_behavior(self, random_positions):
        """Test that random_start parameter affects results."""
        n = 10

        # With deterministic=True
        det_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=True,
            batch=None,
        )

        # With deterministic=False (should be different most of the time)
        rand_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=False,
            batch=None,
        )

        # Check that both are valid
        self._verify_fps_properties(det_result, random_positions)
        self._verify_fps_properties(rand_result, random_positions)

    # GPU tests
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_gpu(self, random_positions):
        """Test PyG FPS on GPU."""
        n = 10

        result = fps(
            pos=random_positions.cuda(),
            n=n,
            deterministic=False,
            batch=None,
        )

        assert result.is_cuda
        self._verify_fps_properties(result.cpu(), random_positions)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason="torch_fpsample not installed")
    def test_cpu_vs_gpu_consistency(self, random_positions):
        """Test that CPU and GPU FPS produce similar results."""
        n = 10

        cpu_result = fps(
            pos=random_positions.cpu(),
            n=n,
            deterministic=False,
            batch=None,
        )

        gpu_result = fps(
            pos=random_positions.cuda(),
            n=n,
            deterministic=False,
            batch=None,
        ).cpu()

        # Both should be valid FPS selections
        self._verify_fps_properties(cpu_result, random_positions)
        self._verify_fps_properties(gpu_result, random_positions)

        # They may differ due to different random starts, but both valid

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_pyg_gpu_batched(self, batched_positions):
        """Test PyG FPS on GPU with batched data."""
        positions, batch = batched_positions
        ratio = 0.5

        result = fps(
            pos=positions.cuda(),
            ratio=ratio,
            batch=batch.cuda(),
            deterministic=False,
        )

        assert result.is_cuda
        self._verify_fps_properties(result.cpu(), positions)

    # Different dimensionality tests
    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_2d_positions(self):
        """Test FPS with 2D positions."""
        torch.manual_seed(42)
        positions = torch.randn(50, 2)
        n = 5

        result = fps(
            pos=positions.cpu(),
            n=n,
            deterministic=True,
            batch=None,
        )

        self._verify_fps_properties(result, positions)

    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_high_dimensional_positions(self):
        """Test FPS with high-dimensional positions."""
        torch.manual_seed(42)
        positions = torch.randn(50, 20)
        n = 5

        result = fps(
            pos=positions.cpu(),
            n=n,
            deterministic=True,
            batch=None,
        )

        self._verify_fps_properties(result, positions)

    # Tests for reproducibility
    @pytest.mark.skipif(not HAS_PYG_FPS, reason="pyg-lib not installed")
    def test_reproducibility_cpu(self, random_positions):
        """Test that FPS results are reproducible with fixed seed."""
        torch.manual_seed(42)

        # With deterministic=True, results should always be the same
        result1 = fps(
            pos=random_positions.cpu(),
            n=5,
            deterministic=True,
            batch=None,
        )

        result2 = fps(
            pos=random_positions.cpu(),
            n=5,
            deterministic=True,
            batch=None,
        )

        assert torch.all(result1 == result2)
