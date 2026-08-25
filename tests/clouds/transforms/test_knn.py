
"""
Equivalence tests for the various kNN backends in the knn module.

These tests do NOT check correctness of the neighbor search (i.e. we don't
verify the "true" nearest neighbors against a brute-force ground truth).
Instead, they check that the different available backends (_pyg_knn,
_keops_knn, _nanoflann_knn) agree with each other on the *set* of neighbor
indices they return for the same input, for both the unbatched and batched
cases, and for both CPU and CUDA tensors where applicable.

Backends whose optional dependency is not installed (pykeops, pynanoflann)
are skipped via pytest.mark.skipif, and CUDA-only tests are skipped when
CUDA is not available.
"""

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

requires_keops = pytest.mark.skipif(not HAS_KEOPS, reason="pykeops is not installed")
requires_nanoflann = pytest.mark.skipif(not HAS_NANOFLANN, reason="pynanoflann is not installed")
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")


def _neighbor_sets(indices: torch.Tensor) -> list[frozenset]:
    """Convert a (N, k) index tensor into a list of per-row frozensets.

    We compare sets rather than raw tensors because different backends may
    return neighbors in different orders, or break ties among equidistant
    points differently.
    """
    return [frozenset(row.tolist()) for row in indices]


def _assert_neighbor_sets_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    sets_a = _neighbor_sets(a)
    sets_b = _neighbor_sets(b)
    mismatches = [i for i, (sa, sb) in enumerate(zip(sets_a, sets_b)) if sa != sb]
    assert not mismatches, (
        f"neighbor sets differ at rows {mismatches[:10]}"
        f"{'...' if len(mismatches) > 10 else ''}"
    )


class TestKNNImplementations:
    """Cross-backend equivalence checks for the kNN implementations."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def unbatched_pos(self):
        torch.manual_seed(0)
        return torch.randn(100, 3)

    @pytest.fixture
    def batched_pos_and_batch(self):
        torch.manual_seed(1)
        # Two "point clouds" of 50 points each, well separated in space so
        # that batching (rather than raw distance) determines membership.
        pos_a = torch.randn(50, 3)
        pos_b = torch.randn(50, 3) + 100.0
        pos = torch.cat([pos_a, pos_b], dim=0)
        batch = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
        return pos, batch

    @pytest.fixture
    def query_subset(self, unbatched_pos):
        # A smaller query set distinct from pos, to exercise query_pos !=
        # pos behavior.
        torch.manual_seed(2)
        return torch.randn(10, 3)

    # ------------------------------------------------------------------
    # PyG vs KeOps (unbatched)
    # ------------------------------------------------------------------

    @requires_keops
    def test_pyg_vs_keops_unbatched_self_query(self, unbatched_pos):
        pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos
        k = 8

        pyg_indices = _pyg_knn(unbatched_pos, k=k)
        keops_indices = _keops_knn(pos_cuda, k=k).cpu()

        _assert_neighbor_sets_equal(pyg_indices, keops_indices)

    @requires_keops
    def test_pyg_vs_keops_unbatched_separate_query(self, unbatched_pos, query_subset):
        pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos
        query_cuda = query_subset.cuda() if CUDA_AVAILABLE else query_subset
        k = 5

        pyg_indices = _pyg_knn(unbatched_pos, k=k, query_pos=query_subset)
        keops_indices = _keops_knn(pos_cuda, k=k, query_pos=query_cuda).cpu()

        _assert_neighbor_sets_equal(pyg_indices, keops_indices)

    @requires_keops
    def test_pyg_vs_keops_batched(self, batched_pos_and_batch):
        pos, batch = batched_pos_and_batch
        pos_cuda = pos.cuda() if CUDA_AVAILABLE else pos
        batch_cuda = batch.cuda() if CUDA_AVAILABLE else batch
        k = 6

        pyg_indices = _pyg_knn(pos, k=k, batch=batch, query_batch=batch)
        keops_indices = _keops_knn(pos_cuda, k=k, batch=batch_cuda, query_batch=batch_cuda).cpu()

        _assert_neighbor_sets_equal(pyg_indices, keops_indices)

        # Sanity check that batching actually mattered: no cross-batch
        # neighbors should appear in either result.
        first_batch_size = int((batch == 0).sum())
        for row in pyg_indices[:first_batch_size]:
            assert (row < first_batch_size).all(), "PyG result leaked cross-batch neighbors"
        for row in keops_indices[:first_batch_size]:
            assert (row < first_batch_size).all(), "KeOps result leaked cross-batch neighbors"

    # ------------------------------------------------------------------
    # PyG vs nanoflann
    # ------------------------------------------------------------------

    @requires_nanoflann
    def test_pyg_vs_nanoflann_unbatched_self_query(self, unbatched_pos):
        k = 8

        pyg_indices = _pyg_knn(unbatched_pos, k=k)
        nanoflann_indices = _nanoflann_knn(unbatched_pos, k=k, query_pos=unbatched_pos)

        _assert_neighbor_sets_equal(pyg_indices, nanoflann_indices)

    @requires_nanoflann
    def test_pyg_vs_nanoflann_unbatched_separate_query(self, unbatched_pos, query_subset):
        k = 5

        pyg_indices = _pyg_knn(unbatched_pos, k=k, query_pos=query_subset)
        nanoflann_indices = _nanoflann_knn(unbatched_pos, k=k, query_pos=query_subset)

        _assert_neighbor_sets_equal(pyg_indices, nanoflann_indices)

    @requires_nanoflann
    def test_pyg_vs_nanoflann_batched(self, batched_pos_and_batch):
        # NOTE: as written, `_nanoflann_knn` does `if query_pos:` when
        # `batch is not None`, which raises for a real (non-empty) tensor
        # query_pos. This test documents/exercises that current behavior.
        pos, batch = batched_pos_and_batch
        k = 6

        pyg_indices = _pyg_knn(pos, k=k, batch=batch, query_batch=batch)

        nanoflann_indices = _nanoflann_knn(pos, k=k, batch=batch, query_pos=pos, query_batch=batch)

        _assert_neighbor_sets_equal(pyg_indices, nanoflann_indices)

        # Sanity check that batching actually mattered: no cross-batch
        # neighbors should appear in either result.
        first_batch_size = int((batch == 0).sum())
        for row in pyg_indices[:first_batch_size]:
            assert (row < first_batch_size).all(), "PyG result leaked cross-batch neighbors"
        for row in nanoflann_indices[:first_batch_size]:
            assert (row < first_batch_size).all(), "KeOps result leaked cross-batch neighbors"
            
    # ------------------------------------------------------------------
    # KeOps vs nanoflann (both optional deps present)
    # ------------------------------------------------------------------

    @requires_keops
    @requires_nanoflann
    def test_keops_vs_nanoflann_unbatched(self, unbatched_pos):
        pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos
        k = 8

        keops_indices = _keops_knn(pos_cuda, k=k).cpu()
        nanoflann_indices = _nanoflann_knn(unbatched_pos, k=k, query_pos=unbatched_pos)

        _assert_neighbor_sets_equal(keops_indices, nanoflann_indices)

    # ------------------------------------------------------------------
    # Distance outputs agree (return_distances=True), where applicable
    # ------------------------------------------------------------------

    @requires_keops
    def test_pyg_vs_keops_distances_unbatched(self, unbatched_pos):
        pos_cuda = unbatched_pos.cuda() if CUDA_AVAILABLE else unbatched_pos
        k = 8

        pyg_dist, pyg_idx = _pyg_knn(unbatched_pos, k=k, query_pos=unbatched_pos, return_distances=True)
        keops_dist, keops_idx = _keops_knn(pos_cuda, k=k, return_distances=True)
        keops_dist, keops_idx = keops_dist.cpu(), keops_idx.cpu()

        _assert_neighbor_sets_equal(pyg_idx, keops_idx)

        # Since both are Euclidean distance and neighbor sets match,
        # sorted per-row distances should match numerically too.
        pyg_sorted, _ = torch.sort(pyg_dist, dim=-1)
        keops_sorted, _ = torch.sort(keops_dist, dim=-1)
        torch.testing.assert_close(pyg_sorted, keops_sorted, rtol=1e-4, atol=1e-4)

    @requires_nanoflann
    def test_pyg_vs_nanoflann_distances_unbatched(self, unbatched_pos):
        k = 8

        pyg_dist, pyg_idx = _pyg_knn(unbatched_pos, k=k, query_pos=unbatched_pos, return_distances=True)
        nf_dist, nf_idx = _nanoflann_knn(
            unbatched_pos, k=k, query_pos=unbatched_pos, return_distances=True
        )

        _assert_neighbor_sets_equal(pyg_idx, nf_idx)

        pyg_sorted, _ = torch.sort(pyg_dist, dim=-1)
        nf_sorted, _ = torch.sort(nf_dist.float(), dim=-1)
        torch.testing.assert_close(pyg_sorted, nf_sorted, rtol=1e-4, atol=1e-4)
