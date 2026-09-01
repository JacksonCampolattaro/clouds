import itertools
import math
import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .apply_selection import apply_selection


class RadiusSelect(BaseTransform):
    def __init__(
        self,
        max_num_points: int,
        max_radius: float = math.inf,
        max_ratio: float = 1.0,
        sort_by_distance: bool = False,
        deterministic: bool = False,
        dims: list[int] | None = None,
    ):
        super().__init__()
        self.max_radius = max_radius
        self.max_num_points = max_num_points
        self.max_ratio = max_ratio
        self.sort_by_distance = sort_by_distance
        self.deterministic = deterministic
        self.dims = dims

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)
        pos = data.pos if self.dims is None else data.pos[:, self.dims]

        if isinstance(data.batch, Tensor):
            assert isinstance(data.ptr, Tensor)
            item_selections = [
                start + self._select_single(pos[start:end, :])  #
                for start, end in itertools.pairwise(data.ptr)
            ]
            data.selection_index = torch.cat(item_selections, dim=0)
        else:
            data.selection_index = self._select_single(pos)

        return data

    def _select_single(self, pos: Tensor) -> Tensor:
        num_points = min(int(pos.size(0) * self.max_ratio), self.max_num_points)
        if pos.size(0) == num_points and not math.isfinite(self.max_radius):
            return torch.arange(num_points, device=pos.device)

        # Select a point at random
        center = pos[random.randrange(0, pos.size(0)), :]
        if self.deterministic:
            center = pos[0, :]

        distances = torch.linalg.vector_norm(pos - center, dim=-1)
        index = distances.argsort()[:num_points]
        if not self.sort_by_distance:
            # TODO: pyg's index_sort should be faster here!
            index, _ = index.sort()

        # If a maximum radius is specified, drop points outside
        if math.isfinite(self.max_radius):
            index = index[distances[index] < self.max_radius]

        return index

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_num_points={self.max_num_points}, "
            f"max_radius={self.max_radius}, "
            f"max_ratio={self.max_ratio}, "
            f"sort_by_distance={self.sort_by_distance}, "
            f"deterministic={self.deterministic}, "
            f"dims={self.dims})"
        )


SphereSelect = RadiusSelect


class SphereCrop(SphereSelect):
    def forward(self, data: Data) -> Data:
        return apply_selection(super().forward(data))
