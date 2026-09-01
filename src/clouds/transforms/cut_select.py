import itertools

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
        assert isinstance(data.pos, Tensor)
        pos = data.pos if self.dims is None else data.pos[:, self.dims]
        
        if isinstance(data.batch, Tensor):
            assert hasattr(data, 'ptr') and isinstance(data.ptr, Tensor)
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
        if num_points == pos.size(0):
            return torch.arange(num_points, device=pos.device)

        # Select a vector at random
        vec = torch.randn([pos.size(-1)], device=pos.device)

        # Dot product is proportional to length along vector
        dot = torch.linalg.vecdot(pos, vec.unsqueeze(0))

        index = dot.argsort()[:num_points]
        if not self.sort_by_distance:
            # TODO: pyg's index_sort would be faster here!
            index, _ = index.sort()

        return index

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
