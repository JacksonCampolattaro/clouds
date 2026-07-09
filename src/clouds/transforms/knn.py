import warnings
from functools import lru_cache

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn as _pyg_knn
from torch_geometric.transforms import BaseTransform

from clouds.data import SourceIndexedData

try:
    HAS_NANOFLANN = True
    from pynanoflann import KDTree
except ImportError:
    HAS_NANOFLANN = False

try:
    HAS_KEOPS = True
    from pykeops.torch import LazyTensor
except ImportError:
    HAS_KEOPS = False


def _diagonal_ranges(batch_x: Tensor = None, batch_y: Tensor = None):
    """Encodes the block-diagonal structure associated to a batch vector."""
    # See: https://github.com/getkeops/keops/issues/73

    b = max(batch_x.amax(), batch_y.amax()) + 1

    def ranges_slices(batch):
        """Helper function for the diagonal ranges function."""
        Ns = batch.bincount(minlength=b)
        indices = Ns.cumsum(0)
        ranges = torch.cat((0 * indices[:1], indices))
        ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
        slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
        return ranges, slices

    if batch_x is None and batch_y is None:
        return None
    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)
    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


@torch.compiler.disable(recursive=True)
@torch.amp.autocast('cuda', enabled=False)
def _keops_knn(
    pos: Tensor,
    batch: Tensor | None = None,
    k: int = 20,
    query_pos: Tensor | None = None,
    query_batch: Tensor | None = None,
    return_distances: bool = False,
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    query_pos = pos if query_pos is None else query_pos
    query_batch = batch if query_batch is None else query_batch
    p_i = LazyTensor(pos.float().unsqueeze(-3))
    p_j = LazyTensor(query_pos.float().unsqueeze(-2))
    d_ij = ((p_i - p_j) ** 2).sum(-1)
    if batch is not None:
        d_ij.ranges = _diagonal_ranges(query_batch, batch)
    indices = d_ij.argKmin(k, dim=1, **kwargs).long()
    if return_distances:
        distances = torch.linalg.vector_norm(query_pos[:, None, :] - pos[indices, :], dim=-1)
        return (distances, indices)
    else:
        return indices


@lru_cache(maxsize=4)
def _cached_kdtree(pos: Tensor):
    kdtree = KDTree()
    kdtree.fit(pos.numpy())
    return kdtree


def _nanoflann_knn(
    pos: Tensor,
    k: int = 20,
    query_pos: Tensor | None = None,
    num_threads: int = 4,
    return_distances: bool = False,
) -> Tensor:
    kdtree = _cached_kdtree(pos)
    distances, indices = kdtree.kneighbors(
        query_pos.numpy(),
        n_neighbors=k,
        n_jobs=num_threads,
    )
    indices = torch.from_numpy(indices).long()
    return (torch.from_numpy(distances), indices) if return_distances else indices


def knn(
    pos: Tensor,
    k: int,
    batch: Tensor | None = None,
    query_pos: Tensor | None = None,
    query_batch: Tensor | None = None,
    num_threads: int = 4,
    return_distances: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    query_pos = pos if query_pos is None else query_pos
    query_batch = batch if query_batch is None else query_batch

    if pos.size(0) < k:
        raise ValueError(f"Attempted to build KNN edges with fewer than K points ({pos.size(0)} < {k})")

    if query_pos.size(0) < 1:
        raise ValueError("Attempted to find neighbors of 0 nodes")

    if pos.is_cuda and HAS_KEOPS:
        return _keops_knn(
            pos,
            k=k,
            batch=batch,
            query_pos=query_pos,
            query_batch=query_batch,
            return_distances=return_distances,
        )
    elif batch is None and HAS_NANOFLANN:
        return _nanoflann_knn(
            pos,
            k=k,
            query_pos=query_pos,
            num_threads=num_threads,
            return_distances=return_distances,
        )
    else:
        warnings.warn("Falling back to PyG's knn query, performance may be poor", stacklevel=2)
        indices = _pyg_knn(
            x=pos,
            y=pos if query_pos is None else query_pos,
            k=k,
            batch_x=batch,
            batch_y=batch if query_batch is None else query_batch,
            num_workers=num_threads,
        )[1].reshape(-1, k)
        if return_distances:
            distances = torch.linalg.vector_norm(query_pos[:, None, :] - pos[indices, :], dim=-1)
            return (distances, indices)
        else:
            return indices


class KNNSourceGraph(BaseTransform):
    def __init__(
        self,
        k=25,
        num_threads: int = 4,
    ):
        self.k = k
        self.num_threads = num_threads

    def forward(self, data: Data) -> Data:
        data.edge_index = knn(
            data.pos,
            k=self.k,
            batch=data.get('batch', None),
            num_threads=self.num_threads,
        ).long()

        return SourceIndexedData(**data.to_dict())
