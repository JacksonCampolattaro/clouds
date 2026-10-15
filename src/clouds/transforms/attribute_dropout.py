import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AttributeDropout(BaseTransform):
    def __init__(self, feature: str, p: float = 0.2) -> None:
        super().__init__()
        self.feature = feature
        self.p = p

    def forward(self, data: Data) -> Data:
        for store in data.node_stores:
            feature = store.get(self.feature)
            if not isinstance(feature, Tensor):
                continue

            if isinstance(getattr(store, 'batch', None), Tensor):
                batch_size = store.batch_size if hasattr(store, 'batch_size') else store.batch.amax() + 1
                mask = torch.rand([batch_size], device=store.batch.device) < self.p
                keep = ~mask[store.batch].reshape(-1, *([1] * (feature.dim() - 1)))
                store[self.feature] = torch.where(keep, feature, torch.zeros_like(feature))
            elif random.random() < self.p:
                store[self.feature] = torch.zeros_like(feature)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.feature}, p={self.p})"
