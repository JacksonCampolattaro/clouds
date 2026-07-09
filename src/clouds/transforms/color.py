import random

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomColorAutoContrast(BaseTransform):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def forward(self, data: Data) -> Data:
        # TODO: support batches?
        assert not isinstance(data.batch, Tensor)

        if random.random() > self.p:
            return data

        colmin = data.color.amin(dim=0, keepdim=True)
        colmax = data.color.amax(dim=0, keepdim=True)
        scale = 1 / (1e-7 + colmax - colmin)
        alpha = self.blend_factor or random.random()
        data.color = (1 - alpha + alpha * scale) * data.color - alpha * colmin * scale

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
