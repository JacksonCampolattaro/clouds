import itertools
import math

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.typing import WITH_KNN as HAS_PYG_KNN

from clouds.transforms.density import EstimateDensity, InverseDensitySelect, _unit_ball_volume


@pytest.fixture
def make_uniform_data():
    def _make_uniform_data(num_points, side_length, dim, seed):
        # A generator-local seed keeps runs reproducible without touching global RNG state.
        generator = torch.Generator().manual_seed(seed)
        return Data(pos=torch.rand(num_points, dim, generator=generator) * side_length)

    return _make_uniform_data


def test_unit_ball_volume_known_values():
    assert _unit_ball_volume(1) == pytest.approx(2.0)
    assert _unit_ball_volume(2) == pytest.approx(math.pi)
    assert _unit_ball_volume(3) == pytest.approx(4.0 / 3.0 * math.pi)


def test_default_configuration():
    transform = EstimateDensity()

    assert transform.pointwise is True
    assert transform.estimation_factor == pytest.approx(0.05)
    assert transform.k == 15
    assert transform.d == 2
    assert transform.V_d == pytest.approx(math.pi)


def test_forward_requires_pos_attribute():
    with pytest.raises(AssertionError):
        EstimateDensity()(Data())


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_pointwise_density_single_cloud_matches_known_density(make_uniform_data):
    num_points, side_length = 6000, 100.0

    torch.manual_seed(1)
    out = EstimateDensity(pointwise=True, estimation_factor=0.05, d=2)(
        make_uniform_data(num_points, side_length, dim=2, seed=1)
    )

    assert out.density.shape == (num_points, 1)
    assert out.density.mean().item() == pytest.approx(num_points / side_length**2, rel=0.2)


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_global_density_single_cloud_matches_known_density(make_uniform_data):
    num_points, side_length = 6000, 100.0

    torch.manual_seed(2)
    out = EstimateDensity(pointwise=False, estimation_factor=0.05, d=2)(
        make_uniform_data(num_points, side_length, dim=2, seed=2)
    )

    # Non-pointwise on an un-batched cloud pools everything into a single scalar estimate.
    assert out.density.numel() == 1
    assert out.density.reshape(-1)[0].item() == pytest.approx(num_points / side_length**2, rel=0.2)


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_pointwise_density_batch_recovers_each_graph_density(make_uniform_data):
    n1, n2, side_length = 4000, 8000, 100.0

    batch = Batch.from_data_list(
        [
            make_uniform_data(n1, side_length, dim=2, seed=10),
            make_uniform_data(n2, side_length, dim=2, seed=11),
        ]
    )
    torch.manual_seed(10)
    out = EstimateDensity(pointwise=True, estimation_factor=0.05, d=2)(batch)

    assert out.density.shape == (n1 + n2, 1)
    assert out.density[batch.batch == 0].mean().item() == pytest.approx(n1 / side_length**2, rel=0.2)
    assert out.density[batch.batch == 1].mean().item() == pytest.approx(n2 / side_length**2, rel=0.2)


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_global_density_batch_recovers_each_graph_density(make_uniform_data):
    n1, n2, side_length = 4000, 8000, 100.0

    batch = Batch.from_data_list(
        [
            make_uniform_data(n1, side_length, dim=2, seed=20),
            make_uniform_data(n2, side_length, dim=2, seed=21),
        ]
    )
    torch.manual_seed(20)
    out = EstimateDensity(pointwise=False, estimation_factor=0.05, d=2)(batch)

    assert out.density.numel() == 2
    assert out.density.reshape(-1)[0].item() == pytest.approx(n1 / side_length**2, rel=0.35)
    assert out.density.reshape(-1)[1].item() == pytest.approx(n2 / side_length**2, rel=0.35)


@pytest.mark.skipif(not HAS_PYG_KNN, reason='PyG kNN not installed')
def test_global_density_3d_matches_known_density(make_uniform_data):
    num_points, side_length, d = 6000, 20.0, 3

    torch.manual_seed(31)
    out = EstimateDensity(pointwise=False, estimation_factor=0.05, d=d)(
        make_uniform_data(num_points, side_length, dim=d, seed=31)
    )

    assert out.density.reshape(-1)[0].item() == pytest.approx(num_points / side_length**d, rel=0.35)


def test_inverse_density_select_initialization():
    assert isinstance(InverseDensitySelect(), InverseDensitySelect)


def test_inverse_density_select_repr():
    repr_str = repr(InverseDensitySelect())
    assert 'InverseDensitySelect' in repr_str
    assert '()' in repr_str


def test_inverse_density_select_forward_single_batch():
    num_points = 100

    selection_index = InverseDensitySelect()(
        Data(pos=torch.randn(num_points, 3), density=torch.rand(num_points, 1) + 0.1)
    ).selection_index

    assert selection_index.shape == (num_points,)
    assert torch.sort(selection_index)[0].tolist() == list(range(num_points))


def test_inverse_density_select_forward_multi_batch():
    num_points1, num_points2 = 30, 40
    ptr = torch.tensor([0, num_points1, num_points1 + num_points2])
    batch = torch.cat([torch.zeros(num_points1), torch.ones(num_points2)]).long()
    data = Data(
        pos=torch.randn(num_points1 + num_points2, 3),
        density=torch.rand(num_points1 + num_points2, 1) + 0.1,
        batch=batch,
        ptr=ptr,
    )

    selection_index = InverseDensitySelect()(data).selection_index
    selected_batch = batch[selection_index]

    assert selection_index.shape == (num_points1 + num_points2,)
    for b, (start, end) in enumerate(itertools.pairwise(ptr)):
        assert torch.sort(selection_index[selected_batch == b])[0].tolist() == list(range(start, end))


def test_inverse_density_selection_weights():
    num_points = 20
    density = torch.ones(num_points, 1)
    density[:10] = 10.0

    first_half = InverseDensitySelect()(Data(pos=torch.randn(num_points, 3), density=density)).selection_index[:10]

    # Low-density points have higher inverse weight, so at least one should be selected early.
    assert any(idx in torch.where(density.flatten() < 2.0)[0] for idx in first_half)
