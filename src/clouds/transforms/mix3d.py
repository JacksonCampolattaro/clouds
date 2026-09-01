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
        batch_size = data.ptr.size(0) - 1

        # Boundaries between items will be deleted with probability 
        new_ptr = data.ptr[
            torch.cat(
                [
                    data.ptr.new_ones(1, dtype=torch.bool),
                    self._boundary_mask(batch_size - 1, data.ptr.device),
                    data.ptr.new_ones(1, dtype=torch.bool),
                ]
            )
        ]

        # Determine which batches are merged-into
        kept_batches = data.batch[new_ptr[:-1]]

        # Copy data into a new Data object
        out = type(data)()
        for key, item in data.items():
            if key == 'ptr':
                out[key] = new_ptr
            elif key == 'batch':
                out[key] = torch.repeat_interleave(
                    torch.arange(new_ptr.size(0) - 1, device=new_ptr.device),
                    new_ptr[1:] - new_ptr[:-1],
                )
            elif item.size(0) == batch_size:
                out[key] = item[kept_batches]
            else:
                out[key] = item

        return out

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
