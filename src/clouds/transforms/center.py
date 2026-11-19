from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation, MinAggregation
from torch_geometric.transforms import BaseTransform


class CenterPoints(BaseTransform):
    def __init__(self, dims: list[int] | None = None) -> None:
        super().__init__()
        self.dims = dims or [0, 1, 2]

    def _aggregate(self, pos: Tensor, index: Tensor | None, ptr: Tensor | None) -> Tensor:
        # PyG reduces in the input dtype, and a float32 mean of large coordinates
        # (e.g. UTM metres) is off by hundreds of metres, so reduce in float64.
        return MeanAggregation()(pos.double(), index=index, ptr=ptr, dim=0)

    def forward(self, data: Data) -> Data:
        offset = None
        for store in data.node_stores:
            pos = store.get('pos')
            if not isinstance(pos, Tensor):
                continue

            batch = getattr(store, 'batch', None)
            if offset is None:
                offset = self._aggregate(pos[:, self.dims], batch, getattr(store, 'ptr', None))

            # Assign a fresh tensor instead of writing through `pos`: cached datasets
            # hand out views into shared storage, and an in-place write there is then
            # copied-on-write by every forked DataLoader worker.
            store.pos = pos.clone()
            offset_per_node = offset[batch] if isinstance(batch, Tensor) else offset
            store.pos[:, self.dims] = (pos[:, self.dims].double() - offset_per_node).to(pos.dtype)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dims={self.dims})"


class GroundPoints(CenterPoints):
    def _aggregate(self, pos: Tensor, index: Tensor | None, ptr: Tensor | None) -> Tensor:
        return MinAggregation()(pos, index=index, ptr=ptr, dim=0)
