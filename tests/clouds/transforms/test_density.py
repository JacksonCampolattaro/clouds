import math

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

from clouds.transforms.density import EstimateDensity, InverseDensitySelect, _unit_ball_volume


def _uniform_positions(num_points: int, side_length: float, dim: int, seed: int) -> torch.Tensor:
    """Uniformly distributed points in [0, side_length]^dim with a fixed,
    generator-local seed (does not disturb global RNG state)."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(num_points, dim, generator=g) * side_length


def _make_uniform_data(num_points: int, side_length: float, dim: int, seed: int) -> Data:
    return Data(pos=_uniform_positions(num_points, side_length, dim, seed))


def _true_density(num_points: int, side_length: float, dim: int) -> float:
    return num_points / (side_length ** dim)


class TestEstimateDensity:
    def setup_method(self):
        # Reseed global RNG before every test so that any internal
        # randomness (e.g. inside `RandomSample`, which likely relies on
        # torch's global generator) is reproducible run-to-run.
        torch.manual_seed(0)

    # -- sanity / config checks, no kNN required -------------------------- #

    def test_unit_ball_volume_known_values(self):
        assert _unit_ball_volume(1) == pytest.approx(2.0)
        assert _unit_ball_volume(2) == pytest.approx(math.pi)
        assert _unit_ball_volume(3) == pytest.approx(4.0 / 3.0 * math.pi)

    def test_default_configuration(self):
        transform = EstimateDensity()
        assert transform.pointwise is True
        assert transform.estimation_factor == pytest.approx(0.05)
        assert transform.k == 15
        assert transform.d == 2
        assert transform.V_d == pytest.approx(math.pi)

    def test_forward_requires_pos_attribute(self):
        data = Data()  # no `pos` set
        transform = EstimateDensity()
        with pytest.raises(AssertionError):
            transform(data)

    # -- single cloud, pointwise vs. global -------------------------------- #

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_pointwise_density_single_cloud_matches_known_density(self):
        num_points, side_length = 6000, 100.0
        true_density = _true_density(num_points, side_length, dim=2)

        data = _make_uniform_data(num_points, side_length, dim=2, seed=1)
        torch.manual_seed(1)
        out = EstimateDensity(pointwise=True, estimation_factor=0.05, d=2)(data)

        assert out.density.shape == (num_points, 1)
        estimated_density = out.density.mean().item()

        # Averaged over thousands of points, so allow moderate error but not
        # an order of magnitude.
        assert estimated_density == pytest.approx(true_density, rel=0.2)

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_global_density_single_cloud_matches_known_density(self):
        num_points, side_length = 6000, 100.0
        true_density = _true_density(num_points, side_length, dim=2)

        data = _make_uniform_data(num_points, side_length, dim=2, seed=2)
        torch.manual_seed(2)
        out = EstimateDensity(pointwise=False, estimation_factor=0.05, d=2)(data)

        # Non-pointwise on an un-batched cloud pools everything into a
        # single scalar estimate.
        assert out.density.numel() == 1
        estimated_density = out.density.reshape(-1)[0].item()

        assert estimated_density == pytest.approx(true_density, rel=0.2)

    # -- batched clouds, pointwise vs. global ------------------------------ #

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_pointwise_density_batch_recovers_each_graph_density(self):
        n1, l1 = 4000, 100.0  # true density 0.4
        n2, l2 = 8000, 100.0  # true density 0.8
        density1 = _true_density(n1, l1, dim=2)
        density2 = _true_density(n2, l2, dim=2)

        data1 = _make_uniform_data(n1, l1, dim=2, seed=10)
        data2 = _make_uniform_data(n2, l2, dim=2, seed=11)
        batch = Batch.from_data_list([data1, data2])

        torch.manual_seed(10)
        out = EstimateDensity(pointwise=True, estimation_factor=0.05, d=2)(batch)

        assert out.density.shape == (n1 + n2, 1)

        est_density1 = out.density[batch.batch == 0].mean().item()
        est_density2 = out.density[batch.batch == 1].mean().item()

        assert est_density1 == pytest.approx(density1, rel=0.2)
        assert est_density2 == pytest.approx(density2, rel=0.2)

    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_global_density_batch_recovers_each_graph_density(self):
        n1, l1 = 4000, 100.0  # true density 0.4
        n2, l2 = 8000, 100.0  # true density 0.8
        density1 = _true_density(n1, l1, dim=2)
        density2 = _true_density(n2, l2, dim=2)

        data1 = _make_uniform_data(n1, l1, dim=2, seed=20)
        data2 = _make_uniform_data(n2, l2, dim=2, seed=21)
        batch = Batch.from_data_list([data1, data2])

        torch.manual_seed(20)
        out = EstimateDensity(pointwise=False, estimation_factor=0.05, d=2)(batch)

        assert out.density.numel() == 2

        est_density1 = out.density.reshape(-1)[0].item()
        est_density2 = out.density.reshape(-1)[1].item()

        assert est_density1 == pytest.approx(density1, rel=0.35)
        assert est_density2 == pytest.approx(density2, rel=0.35)


    @pytest.mark.skipif(not HAS_PYG_KNN, reason="PyG kNN not installed")
    def test_global_density_3d_matches_known_density(self):
        num_points, side_length, d = 6000, 20.0, 3
        true_density = _true_density(num_points, side_length, dim=d)

        data = _make_uniform_data(num_points, side_length, dim=d, seed=31)
        torch.manual_seed(31)
        out = EstimateDensity(pointwise=False, estimation_factor=0.05, d=d)(data)

        estimated_density = out.density.reshape(-1)[0].item()

        assert estimated_density == pytest.approx(true_density, rel=0.35)


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
