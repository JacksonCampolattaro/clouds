import pytest
import torch

from clouds.transforms.knn import (
    HAS_KEOPS,
    HAS_NANOFLANN,
    _keops_knn,
    _nanoflann_knn,
    _pyg_knn,
)

CUDA_AVAILABLE = torch.cuda.is_available()

requires_keops = pytest.mark.skipif(not HAS_KEOPS, reason='pykeops is not installed')
requires_nanoflann = pytest.mark.skipif(not HAS_NANOFLANN, reason='pynanoflann is not installed')


@pytest.fixture
def unbatched_pos():
    torch.manual_seed(0)
    return torch.randn(100, 3)


@pytest.fixture
def batched_pos_and_batch():
    torch.manual_seed(1)
    # Two clouds, well separated so that batching (not distance) determines membership.
    pos = torch.cat([torch.randn(50, 3), torch.randn(50, 3) + 100.0], dim=0)
    batch = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    return pos, batch


@pytest.fixture
def query_subset(unbatched_pos):
    torch.manual_seed(2)
    return torch.randn(10, 3)


@requires_keops
def test_pyg_vs_keops_unbatched_self_query(unbatched_pos):
    pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos

    pyg_indices = _pyg_knn(unbatched_pos, k=8)
    keops_indices = _keops_knn(pos_cuda, k=8).cpu()

    assert pyg_indices.shape == keops_indices.shape
    assert [set(row.tolist()) for row in pyg_indices] == [set(row.tolist()) for row in keops_indices]


@requires_keops
def test_pyg_vs_keops_unbatched_separate_query(unbatched_pos, query_subset):
    pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos
    query_cuda = query_subset.cuda() if CUDA_AVAILABLE else query_subset

    pyg_indices = _pyg_knn(unbatched_pos, k=5, query_pos=query_subset)
    keops_indices = _keops_knn(pos_cuda, k=5, query_pos=query_cuda).cpu()

    assert pyg_indices.shape == keops_indices.shape
    assert [set(row.tolist()) for row in pyg_indices] == [set(row.tolist()) for row in keops_indices]


@requires_keops
def test_pyg_vs_keops_batched(batched_pos_and_batch):
    pos, batch = batched_pos_and_batch
    pos_cuda = pos.cuda() if CUDA_AVAILABLE else pos
    batch_cuda = batch.cuda() if CUDA_AVAILABLE else batch

    pyg_indices = _pyg_knn(pos, k=6, batch=batch, query_batch=batch)
    keops_indices = _keops_knn(pos_cuda, k=6, batch=batch_cuda, query_batch=batch_cuda).cpu()

    assert pyg_indices.shape == keops_indices.shape
    assert [set(row.tolist()) for row in pyg_indices] == [set(row.tolist()) for row in keops_indices]

    first_batch_size = int((batch == 0).sum())
    for row in pyg_indices[:first_batch_size]:
        assert (row < first_batch_size).all()
    for row in keops_indices[:first_batch_size]:
        assert (row < first_batch_size).all()


@requires_nanoflann
def test_pyg_vs_nanoflann_unbatched_self_query(unbatched_pos):
    pyg_indices = _pyg_knn(unbatched_pos, k=8)
    nanoflann_indices = _nanoflann_knn(unbatched_pos, k=8, query_pos=unbatched_pos)

    assert pyg_indices.shape == nanoflann_indices.shape
    assert [set(row.tolist()) for row in pyg_indices] == [set(row.tolist()) for row in nanoflann_indices]


@requires_nanoflann
def test_pyg_vs_nanoflann_unbatched_separate_query(unbatched_pos, query_subset):
    pyg_indices = _pyg_knn(unbatched_pos, k=5, query_pos=query_subset)
    nanoflann_indices = _nanoflann_knn(unbatched_pos, k=5, query_pos=query_subset)

    assert pyg_indices.shape == nanoflann_indices.shape
    assert [set(row.tolist()) for row in pyg_indices] == [set(row.tolist()) for row in nanoflann_indices]


@requires_nanoflann
def test_pyg_vs_nanoflann_batched(batched_pos_and_batch):
    pos, batch = batched_pos_and_batch

    pyg_indices = _pyg_knn(pos, k=6, batch=batch, query_batch=batch)
    nanoflann_indices = _nanoflann_knn(pos, k=6, batch=batch, query_pos=pos, query_batch=batch)

    assert pyg_indices.shape == nanoflann_indices.shape
    assert [set(row.tolist()) for row in pyg_indices] == [set(row.tolist()) for row in nanoflann_indices]

    first_batch_size = int((batch == 0).sum())
    for row in pyg_indices[:first_batch_size]:
        assert (row < first_batch_size).all()
    for row in nanoflann_indices[:first_batch_size]:
        assert (row < first_batch_size).all()


@requires_keops
@requires_nanoflann
def test_keops_vs_nanoflann_unbatched(unbatched_pos):
    pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos

    keops_indices = _keops_knn(pos_cuda, k=8).cpu()
    nanoflann_indices = _nanoflann_knn(unbatched_pos, k=8, query_pos=unbatched_pos)

    assert keops_indices.shape == nanoflann_indices.shape
    assert [set(row.tolist()) for row in keops_indices] == [set(row.tolist()) for row in nanoflann_indices]


@requires_keops
def test_pyg_vs_keops_distances_unbatched(unbatched_pos):
    pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos

    pyg_dist, pyg_idx = _pyg_knn(unbatched_pos, k=8, query_pos=unbatched_pos, return_distances=True)
    keops_dist, keops_idx = _keops_knn(pos_cuda, k=8, return_distances=True)
    keops_dist, keops_idx = keops_dist.cpu(), keops_idx.cpu()

    assert [set(row.tolist()) for row in pyg_idx] == [set(row.tolist()) for row in keops_idx]
    torch.testing.assert_close(torch.sort(pyg_dist, dim=-1)[0], torch.sort(keops_dist, dim=-1)[0], rtol=1e-4, atol=1e-4)


@requires_nanoflann
def test_pyg_vs_nanoflann_distances_unbatched(unbatched_pos):
    pyg_dist, pyg_idx = _pyg_knn(unbatched_pos, k=8, query_pos=unbatched_pos, return_distances=True)
    nf_dist, nf_idx = _nanoflann_knn(unbatched_pos, k=8, query_pos=unbatched_pos, return_distances=True)

    assert [set(row.tolist()) for row in pyg_idx] == [set(row.tolist()) for row in nf_idx]
    torch.testing.assert_close(torch.sort(pyg_dist, dim=-1)[0], torch.sort(nf_dist.float(), dim=-1)[0], rtol=1e-4, atol=1e-4)
