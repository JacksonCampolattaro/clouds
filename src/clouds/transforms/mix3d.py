from random import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Mix3D(BaseTransform):
    def __init__(self, p: float = 0.8) -> None:
        super().__init__()
        self.p = p

    def forward(self, data: Data) -> Data:
        assert isinstance(data.ptr, Tensor)

        # Boundaries between items will be deleted with probability p
        data.ptr = data.ptr[
            torch.cat(
                [
                    data.ptr.new_ones(1, dtype=torch.bool),
                    self._boundary_mask(data.ptr.size(0) - 2, data.ptr.device),
                    data.ptr.new_ones(1, dtype=torch.bool),
                ]
            )
        ]

        # Replace data.batch to match new data.ptr
        data.batch = torch.repeat_interleave(
            torch.arange(data.ptr.size(0) - 1, device=data.ptr.device),
            data.ptr[1:] - data.ptr[:-1],
        )

        data.batch_size = data.ptr.size(0)

        return data

    def _boundary_mask(self, n: int, device: torch.device) -> Tensor:
        """Sample which interior boundaries to delete, never deleting
        two adjacent ones (so at most 2 items ever get merged)."""
        keep = torch.ones(n, dtype=torch.bool, device=device)
        skip_next = False
        for i in range(n):
            if skip_next:
                skip_next = False
                continue
            if random() < self.p:  # delete this boundary
                keep[i] = False
                skip_next = True  # force-keep the next one
        return keep
