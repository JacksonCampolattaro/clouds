from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation, MinAggregation
from torch_geometric.transforms import BaseTransform


class CenterPoints(BaseTransform):
    def __init__(self, dims=None):
        super().__init__()
        self.dims = dims or [0, 1, 2]

    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            if not isinstance(store.pos, Tensor):
                continue

            if offset is None:
                offset = MeanAggregation()(
                    store.pos[:, self.dims],
                    index=getattr(store, 'batch', None),
                    ptr=getattr(store, 'ptr', None),
                    dim=0,
                )

            if isinstance(store.batch, Tensor):
                store.pos[:, self.dims] = store.pos[:, self.dims] - offset[store.batch, None]
            else:
                store.pos[:, self.dims] = store.pos[:, self.dims] - offset

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class GroundPoints(CenterPoints):
    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            if not isinstance(store.pos, Tensor):
                continue

            if offset is None:
                offset = MinAggregation()(
                    store.pos[:, self.dims],
                    index=getattr(store, 'batch', None),
                    ptr=getattr(store, 'ptr', None),
                    dim=0,
                )

            if isinstance(store.batch, Tensor):
                store.pos[:, self.dims] = store.pos[:, self.dims] - offset[store.batch, None]
            else:
                store.pos[:, self.dims] = store.pos[:, self.dims] - offset

        return data
