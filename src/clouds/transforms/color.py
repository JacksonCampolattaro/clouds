import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MaxAggregation, MinAggregation
from torch_geometric.transforms import BaseTransform


class RandomColorAutoContrast(BaseTransform):
    def __init__(self, p: float = 0.2, blend_factor: float | None = None) -> None:
        self.p = p
        self.blend_factor = blend_factor

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            color = store.get('color')
            if not isinstance(color, Tensor):
                continue

            batch = getattr(store, 'batch', None)
            if not isinstance(batch, Tensor) and random.random() > self.p:
                continue

            colmin = MinAggregation()(color, index=batch, ptr=getattr(store, 'ptr', None), dim=0)
            colmax = MaxAggregation()(color, index=batch, ptr=getattr(store, 'ptr', None), dim=0)
            scale = 1 / (1e-7 + colmax - colmin)
            alpha = self.blend_factor if self.blend_factor is not None else torch.rand_like(scale)
            if isinstance(batch, Tensor):
                if self.p != 1.0:
                    raise NotImplementedError()
                colmin, scale = colmin[batch], scale[batch]
                if isinstance(alpha, Tensor):
                    alpha = alpha[batch]
            store.color = (1 - alpha + alpha * scale) * color - alpha * colmin * scale

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
