import pytest
import torch
from torch_geometric.typing import WITH_FPS as HAS_PYG_FPS

from clouds.transforms.fps import HAS_TORCH_FPSAMPLE, fps


@pytest.fixture
def random_positions():
    torch.manual_seed(42)
    return torch.randn(100, 3)


@pytest.fixture
def small_positions():
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
        ]
    )


@pytest.fixture
def clustered_positions():
    torch.manual_seed(42)
    return torch.cat([torch.randn(30, 3) * 0.1, torch.randn(30, 3) * 0.1 + 10.0], dim=0)


@pytest.fixture
def batched_positions():
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
        ]
    )
    return positions, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason='torch_fpsample not installed')
def test_fpsample_deterministic_is_reproducible(random_positions):
    first = fps(pos=random_positions.cpu(), n=10, deterministic=True)
    second = fps(pos=random_positions.cpu(), n=10, deterministic=True)

    assert torch.equal(first, second)
    assert first.min() >= 0
    assert first.max() < random_positions.size(0)
    assert len(torch.unique(first)) == len(first)


@pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason='torch_fpsample not installed')
def test_fpsample_with_ratio(random_positions):
    result = fps(pos=random_positions.cpu(), ratio=0.2, deterministic=True)

    assert len(result) == 20
    assert result.min() >= 0
    assert result.max() < random_positions.size(0)
    assert len(torch.unique(result)) == len(result)


@pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason='torch_fpsample not installed')
def test_fpsample_deterministic_vs_random(random_positions):
    deterministic = fps(pos=random_positions.cpu(), n=10, deterministic=True)
    random_start = fps(pos=random_positions.cpu(), n=10, deterministic=False)

    assert len(deterministic) == len(random_start)
    assert deterministic.min() >= 0
    assert deterministic.max() < random_positions.size(0)
    assert len(torch.unique(deterministic)) == len(deterministic)
    assert random_start.min() >= 0
    assert random_start.max() < random_positions.size(0)
    assert len(torch.unique(random_start)) == len(random_start)


@pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason='torch_fpsample not installed')
def test_fpsample_clustered_data(clustered_positions):
    result = fps(pos=clustered_positions.cpu(), n=4, deterministic=True)

    assert result.min() >= 0
    assert result.max() < clustered_positions.size(0)
    assert len(torch.unique(result)) == len(result)

    selected = clustered_positions[result]
    dist1 = torch.norm(selected - torch.tensor([0.0, 0.0, 0.0]), dim=1)
    dist2 = torch.norm(selected - torch.tensor([10.0, 10.0, 10.0]), dim=1)
    assert (dist1 < dist2).sum() > 0
    assert (dist2 < dist1).sum() > 0


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_cpu(random_positions):
    result = fps(pos=random_positions.cpu(), n=10, deterministic=False, batch=None)

    assert len(result) == 10
    assert result.min() >= 0
    assert result.max() < random_positions.size(0)
    assert len(torch.unique(result)) == len(result)


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_with_ratio(random_positions):
    assert len(fps(pos=random_positions.cpu(), ratio=0.2, deterministic=False, batch=None)) == 20


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_batched(batched_positions):
    positions, batch = batched_positions

    result = fps(pos=positions.cpu(), ratio=0.5, batch=batch, deterministic=False)

    assert len(result[batch[result] == 0]) == 2
    assert len(result[batch[result] == 1]) == 2


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_batched_with_batch_size(batched_positions):
    positions, batch = batched_positions

    result = fps(pos=positions.cpu(), ratio=0.5, batch=batch, batch_size=2, deterministic=False)

    assert result.min() >= 0
    assert result.max() < positions.size(0)
    assert len(torch.unique(result)) == len(result)


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_edge_cases(small_positions):
    single = fps(pos=small_positions.cpu(), n=1, deterministic=False)
    assert len(single) == 1
    assert 0 <= single[0] < len(small_positions)

    full = fps(pos=small_positions.cpu(), n=len(small_positions), deterministic=False)
    assert torch.equal(torch.sort(full)[0], torch.arange(len(small_positions)))


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_random_vs_deterministic_behavior(random_positions):
    deterministic = fps(pos=random_positions.cpu(), n=10, deterministic=True, batch=None)
    random_start = fps(pos=random_positions.cpu(), n=10, deterministic=False, batch=None)

    assert len(deterministic) == len(random_start) == 10
    assert deterministic.min() >= 0
    assert deterministic.max() < random_positions.size(0)
    assert random_start.min() >= 0
    assert random_start.max() < random_positions.size(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_gpu(random_positions):
    result = fps(pos=random_positions.cuda(), n=10, deterministic=False, batch=None)

    assert result.is_cuda
    assert len(result) == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
def test_pyg_gpu_batched(batched_positions):
    positions, batch = batched_positions

    result = fps(pos=positions.cuda(), ratio=0.5, batch=batch.cuda(), deterministic=False)

    assert result.is_cuda
    assert len(result) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
@pytest.mark.skipif(not HAS_TORCH_FPSAMPLE, reason='torch_fpsample not installed')
def test_cpu_vs_gpu_consistency(random_positions):
    cpu_result = fps(pos=random_positions.cpu(), n=10, deterministic=False, batch=None)
    gpu_result = fps(pos=random_positions.cuda(), n=10, deterministic=False, batch=None).cpu()

    assert cpu_result.min() >= 0
    assert cpu_result.max() < random_positions.size(0)
    assert gpu_result.min() >= 0
    assert gpu_result.max() < random_positions.size(0)


@pytest.mark.skipif(not HAS_PYG_FPS, reason='pyg-lib not installed')
@pytest.mark.parametrize('dim', [2, 20])
def test_pyg_dimensionality(dim):
    torch.manual_seed(42)
    positions = torch.randn(50, dim)

    result = fps(pos=positions.cpu(), n=5, deterministic=True, batch=None)

    assert result.min() >= 0
    assert result.max() < positions.size(0)
    assert len(torch.unique(result)) == len(result)
