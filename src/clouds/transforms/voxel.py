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

    def __init__(self, voxel_size: float | tuple[float, float], large_voxel_prob: float = 0.5) -> None:
        self.voxel_size = voxel_size
        self.large_voxel_prob = large_voxel_prob

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)

        # Determine voxel size
        def get_voxel_size() -> float:
            if isinstance(self.voxel_size, tuple):
                if random.random() > self.large_voxel_prob:
                    return self.voxel_size[0]
                return random.uniform(*self.voxel_size)
            return self.voxel_size

        # Create clusters (global IDs, offset per batch)
        cluster = voxel_grid(data.pos, get_voxel_size(), data.batch)

        # Remap to contiguous, sequential IDs (0 .. num_clusters-1)
        # FIXME: do not reorder!
        unique_clusters = torch.unique(cluster)
        data.num_clusters = unique_clusters.size(0)
        data.cluster_index = torch.searchsorted(unique_clusters, cluster)

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
        if data.pos.is_cpu and HAS_VPSAMPLE:
            if isinstance(data.batch, Tensor):
                batch_size = data.batch_size if hasattr(data, 'batch_size') else torch.amax(data.batch) + 1
                voxel_sizes = torch.tensor([get_voxel_size() for _ in range(batch_size)], device=data.pos.device)
                scaled_pos = data.pos * (1 / voxel_sizes)[data.batch, None]
                offset_axis = -1
                max_graph_size = scaled_pos[:, offset_axis].amax() - scaled_pos[:, offset_axis].amin() + 1
                offsets = torch.zeros_like(data.pos)
                offsets[:, offset_axis] = data.batch * max_graph_size

                data.selection_index = (
                    voxel_subsample(
                        scaled_pos + offsets,
                        voxel_size=1.0,
                        # FIXME: broken for deterministic sampling on small point clouds!
                        pick=pick if self.deterministic else None,
                    )
                    .sort()
                    .values
                )
            else:
                data.selection_index = voxel_subsample(
                    data.pos,
                    voxel_size=get_voxel_size(),
                    hash_size=self.hash_size,
                    # FIXME: broken for deterministic sampling on small point clouds!
                    pick=pick if self.deterministic else None,
                )
        else:
            if isinstance(data.batch, Tensor):
                assert not isinstance(self.voxel_size, tuple)
            data = Compose(
                [
                    VoxelCluster(voxel_size=self.voxel_size, large_voxel_prob=self.large_voxel_prob),
                    ClusterSelect(pick=pick if self.deterministic else None),
                ]
            )(data)
            del data.cluster_index

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.voxel_size}, "  #
            + (f"p={self.large_voxel_prob}, " if isinstance(self.voxel_size, tuple) else "")
            + f"deterministic={self.deterministic})"
        )


class VoxelSample(VoxelSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))
