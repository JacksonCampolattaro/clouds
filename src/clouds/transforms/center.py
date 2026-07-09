from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CenterPoints(BaseTransform):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims or [0, 1, 2]

    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            if isinstance(store.pos, Tensor):
                if offset is None:
                    offset = store.pos[:, self.dims].mean(dim=-2, keepdim=True)
                store.pos[:, self.dims] = store.pos[:, self.dims] - offset

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class GroundPoints(CenterPoints):
    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            if isinstance(store.pos, Tensor):
                if offset is None:
                    offset, _ = store.pos[:, self.dims].min(dim=0)
                store.pos[:, self.dims] = store.pos[:, self.dims] - offset

        return data
