import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

from clouds.transforms.density import EstimateDensity, InverseDensitySelect


class TestEstimateDensity:
    def test_initialization(self):
        transform = EstimateDensity()
        assert transform.pointwise is True
        assert transform.estimation_factor == 0.05
        assert transform.k == 15
        assert transform.V_d > 0

        transform_custom = EstimateDensity(pointwise=False, estimation_factor=0.1, d=3)
        assert transform_custom.pointwise is False
        assert transform_custom.estimation_factor == 0.1

    def test_repr(self):
        transform = EstimateDensity()
        repr_str = repr(transform)
        assert "EstimateDensity" in repr_str
        assert "pointwise=True" in repr_str
        assert "estimation_factor=0.05" in repr_str

        transform_custom = EstimateDensity(pointwise=False, estimation_factor=0.1)
        repr_str = repr(transform_custom)
        assert "pointwise=False" in repr_str
        assert "estimation_factor=0.1" in repr_str

    def test_forward_pointwise(self):
        # Create simple point cloud data
        pos = torch.randn(100, 3)
        data = Data(pos=pos)

        transform = EstimateDensity(pointwise=True, estimation_factor=0.2)
        result = transform(data)

        # Check that density was added
        assert hasattr(result, 'density')
        assert isinstance(result.density, torch.Tensor)
        # Pointwise density should have same number of points
        assert result.density.shape == (100, 1)
        # Density values should be positive
        assert (result.density > 0).all()

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_forward_pooled(self):
        # Create batched point cloud data
        pos1 = torch.randn(50, 3)
        pos2 = torch.randn(60, 3)
        pos = torch.cat([pos1, pos2], dim=0)
        batch = torch.cat([torch.zeros(50), torch.ones(60)], dim=0).long()
        data = Batch(pos=pos, batch=batch, ptr=torch.tensor([0, 50, 110]).long())

        transform = EstimateDensity(pointwise=False, estimation_factor=0.2)
        result = transform(data)

        # Check that density was added
        assert hasattr(result, 'density')
        assert isinstance(result.density, torch.Tensor)
        # Pooled density should have one value per batch
        assert result.density.size(0) == 2
        assert (result.density > 0).all()

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_forward_with_batch(self):
        # Test with explicit batch and ptr
        pos = torch.randn(80, 3)
        batch = torch.tensor([0] * 40 + [1] * 40)
        data = Batch(pos=pos, batch=batch, ptr=torch.tensor([0, 40, 80]))

        transform = EstimateDensity(pointwise=True, estimation_factor=0.15)
        result = transform(data)

        assert hasattr(result, 'density')
        assert result.density.shape == (80, 1)
        assert (result.density > 0).all()


class TestInverseDensitySelect:
    def test_initialization(self):
        transform = InverseDensitySelect()
        assert isinstance(transform, InverseDensitySelect)

    def test_repr(self):
        transform = InverseDensitySelect()
        repr_str = repr(transform)
        assert "InverseDensitySelect" in repr_str
        assert "()" in repr_str

    def test_forward_single_batch(self):
        # Create data with density
        num_points = 100
        pos = torch.randn(num_points, 3)
        density = torch.rand(num_points, 1) + 0.1  # Avoid zeros
        data = Data(pos=pos, density=density)

        transform = InverseDensitySelect()
        result = transform(data)

        # Check selection_index was added
        assert hasattr(result, 'selection_index')
        assert isinstance(result.selection_index, torch.Tensor)
        # Should have same number of points
        assert result.selection_index.shape == (num_points,)
        # Should be a permutation of indices
        assert torch.sort(result.selection_index)[0].tolist() == list(range(num_points))

    def test_forward_multi_batch(self):
        # Create batched data
        num_points1, num_points2 = 30, 40
        pos1 = torch.randn(num_points1, 3)
        pos2 = torch.randn(num_points2, 3)
        pos = torch.cat([pos1, pos2], dim=0)

        density1 = torch.rand(num_points1, 1) + 0.1
        density2 = torch.rand(num_points2, 1) + 0.1
        density = torch.cat([density1, density2], dim=0)

        batch = torch.cat([torch.zeros(num_points1), torch.ones(num_points2)], dim=0).long()
        ptr = torch.tensor([0, num_points1, num_points1 + num_points2])
        data = Data(pos=pos, density=density, batch=batch, ptr=ptr)

        transform = InverseDensitySelect()
        result = transform(data)

        # Check selection_index
        assert hasattr(result, 'selection_index')
        assert isinstance(result.selection_index, torch.Tensor)
        assert result.selection_index.shape == (num_points1 + num_points2,)

        # Check that selection is per-batch (should maintain batch structure)
        selected_batch = batch[result.selection_index]
        # Within each batch, indices should be a permutation
        for start, end in zip(ptr[:-1], ptr[1:]):
            batch_indices = result.selection_index[(selected_batch == 0) if start == 0 else (selected_batch == 1)]
            # The sorted batch indices should match the original range
            assert torch.sort(batch_indices)[0].tolist() == list(range(start, end))

    def test_inverse_density_selection_weights(self):
        # Test that points with lower density are selected first
        num_points = 20
        pos = torch.randn(num_points, 3)
        # Create two distinct density groups
        density = torch.ones(num_points, 1)
        density[:10] = 10.0  # High density (low inverse weight)
        density[10:] = 1.0  # Low density (high inverse weight)
        data = Data(pos=pos, density=density)

        transform = InverseDensitySelect()
        result = transform(data)

        # The first selected points should be from low density region
        # (inverse weights are 1/10 vs 1, so low density points have higher selection probability)
        first_half = result.selection_index[:10]
        # At least some of the first selected should be from low density region
        low_density_indices = torch.where(density.flatten() < 2.0)[0]
        assert any(idx in low_density_indices for idx in first_half)
