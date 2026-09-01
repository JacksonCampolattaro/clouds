import random

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MaxAggregation, MinAggregation
from torch_geometric.transforms import BaseTransform


class RandomColorAutoContrast(BaseTransform):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def forward(self, data: Data) -> Data:
        if not isinstance(data.batch, Tensor) and random.random() > self.p:
            return data

        colmin = MinAggregation()(data.color, index=data.batch, ptr=getattr(data, 'ptr', None), dim=0)
        colmax = MaxAggregation()(data.color, index=data.batch, ptr=getattr(data, 'ptr', None), dim=0)
        scale = 1 / (1e-7 + colmax - colmin)
        alpha = self.blend_factor or torch.rand_like(scale)
        if isinstance(data.batch, Tensor):
            scale, alpha = scale[data.batch], alpha[data.batch]

        data.color = (1 - alpha + alpha * scale) * data.color - alpha * colmin * scale
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
