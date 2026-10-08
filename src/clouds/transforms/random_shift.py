import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomShift(BaseTransform):
    def __init__(self, max_offset: float | list[float] | Tensor = 1.0, attr: str = 'pos') -> None:
        super().__init__()
        self.max_offset = torch.tensor(max_offset)
        self.attr = attr

    def forward(self, data: Data) -> Data:
        stores = [store for store in data.node_stores if isinstance(store.get(self.attr), Tensor)]
        if not stores:
            return data

        offset = torch.rand(stores[0][self.attr].size(-1)) * (2 * self.max_offset) - self.max_offset
        for store in stores:
            attribute = store[self.attr]
            store[self.attr] = attribute + offset.unsqueeze(0).to(device=attribute.device)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(max_offset={self.max_offset.tolist()}, attr={self.attr!r})'
