import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomShift(BaseTransform):
    def __init__(self, max_offset: float | list[float] | Tensor = 1.0, attr: str = 'pos'):
        super().__init__()
        self.max_offset = torch.tensor(max_offset)
        self.attr = attr

    def forward(self, data: Data) -> Data:
        offset = None
        if isinstance(data[self.attr], Tensor):
            if offset is None:
                offset = torch.rand(data[self.attr].size(-1)) * (2 * self.max_offset) - self.max_offset

            data[self.attr] += offset.unsqueeze(0).to(device=data[self.attr].device)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(max_offset={self.max_offset.tolist()}, attr={self.attr!r})'
