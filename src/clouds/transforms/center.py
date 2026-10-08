from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation, MinAggregation
from torch_geometric.transforms import BaseTransform


class CenterPoints(BaseTransform):
    def __init__(self, dims: list[int] | None = None) -> None:
        super().__init__()
        self.dims = dims or [0, 1, 2]

    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue

            if offset is None:
                offset = MeanAggregation()(
                    pos[:, self.dims],
                    index=getattr(store, 'batch', None),
                    ptr=getattr(store, 'ptr', None),
                    dim=0,
                )

            batch = getattr(store, 'batch', None)
            if isinstance(batch, Tensor):
                store.pos[:, self.dims] = pos[:, self.dims] - offset[batch]
            else:
                store.pos[:, self.dims] = pos[:, self.dims] - offset

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class GroundPoints(CenterPoints):
    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue

            if offset is None:
                offset = MinAggregation()(
                    pos[:, self.dims],
                    index=getattr(store, 'batch', None),
                    ptr=getattr(store, 'ptr', None),
                    dim=0,
                )

            batch = getattr(store, 'batch', None)
            if isinstance(batch, Tensor):
                store.pos[:, self.dims] = pos[:, self.dims] - offset[batch]
            else:
                store.pos[:, self.dims] = pos[:, self.dims] - offset

        return data
