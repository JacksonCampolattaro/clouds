import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import voxel_grid
from torch_geometric.transforms import BaseTransform, Compose

from .apply_selection import apply_selection
from .cluster import ClusterSelect

try:
    HAS_VPSAMPLE = True
    from vpsample import voxel_subsample
except (ImportError, RuntimeError):
    HAS_VPSAMPLE = False


class VoxelCluster(BaseTransform):
    """Apply voxel clustering to point cloud data."""

    def __init__(self, voxel_size: float | tuple[float, float]) -> None:
        self.voxel_size = voxel_size

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)

        # Determine voxel size
        voxel_size = random.uniform(*self.voxel_size) if isinstance(self.voxel_size, tuple) else self.voxel_size

        # Create clusters (global IDs, offset per batch)
        cluster = voxel_grid(data.pos, voxel_size, data.batch)

        # Remap to contiguous, sequential IDs (0 .. num_clusters-1)
        unique_clusters = torch.unique(cluster)
        cluster = torch.searchsorted(unique_clusters, cluster)

        # Sort and apply selection
        data.selection_index = selection_index = torch.argsort(cluster)
        data = apply_selection(data)
        data.cluster = cluster[selection_index]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.voxel_size})"


class VoxelSelect(BaseTransform):
    """Select voxel subsamples from point cloud data."""

    def __init__(
        self,
        voxel_size: float | tuple[float, float] = 1.0,
        hash_size: float = 1.0,
        deterministic: bool = False,
        pick: int | None = None,
        large_voxel_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.hash_size = hash_size
        self.deterministic = deterministic
        self.pick = pick
        self.current_pick = 0
        self.large_voxel_prob = large_voxel_prob

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)

        # Determine pick value
        pick = self.pick if self.pick is not None else self.current_pick
        self.current_pick = (self.current_pick + 1) % 12  # FIXME: shouldn't be fixed at 12

        # Determine voxel size
        def get_voxel_size() -> float:
            if isinstance(self.voxel_size, tuple):
                if random.random() > self.large_voxel_prob:
                    return self.voxel_size[0]
                return random.uniform(*self.voxel_size)
            return self.voxel_size

        # Apply subsampling
        if data.pos.is_cpu and not isinstance(data.batch, Tensor) and HAS_VPSAMPLE:
            data.selection_index = voxel_subsample(
                data.pos,
                voxel_size=get_voxel_size(),
                hash_size=self.hash_size,
                # FIXME: broken for deterministic sampling on small point clouds!
                pick=pick if self.deterministic else None,
            )
        else:
            data = Compose(
                [
                    VoxelCluster(voxel_size=get_voxel_size()),
                    ClusterSelect(pick=pick if self.deterministic else None),
                ]
            )(data)
            del data.cluster

        return data


class VoxelSample(VoxelSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))
