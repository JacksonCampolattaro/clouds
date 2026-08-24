import math

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .apply_selection import apply_selection


class CutSelect(BaseTransform):
    def __init__(
        self,
        max_num_points: int,
        max_ratio: float = 1.0,
        sort_by_distance: bool = False,
        dims: list[int] | None = None,
    ):
        super().__init__()
        self.max_num_points = max_num_points
        self.max_ratio = max_ratio
        self.sort_by_distance = sort_by_distance
        self.dims = dims

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor) and isinstance(data.num_nodes, int)
        assert not isinstance(data.batch, Tensor)

        pos = data.pos if self.dims is None else data.pos[:, self.dims]

        num_points = min(int(data.num_nodes * self.max_ratio), self.max_num_points)
        if data.num_nodes <= num_points and not math.isfinite(self.max_radius):
            return data

        # Select a vector at random
        vec = torch.randn([pos.size(-1)], device=pos.device)

        # Dot product is proportional to length along vector
        dot = torch.linalg.vecdot(pos, vec.unsqueeze(0))

        data.selection_index = dot.argsort()[:num_points]
        if not self.sort_by_distance:
            # TODO: pyg's index_sort should be faster here!
            data.selection_index, _ = data.selection_index.sort()

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_num_points={self.max_num_points}, "
            f"max_ratio={self.max_ratio}, "
            f"sort_by_distance={self.sort_by_distance}, "
            f"dims={self.dims})"
        )


class CutCrop(CutSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))
