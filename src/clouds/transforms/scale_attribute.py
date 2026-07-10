import random

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class ScaleAttribute(BaseTransform):
    def __init__(self, attribute: str, factor: float, p: float = 1.0):
        super().__init__()
        self.attribute = attribute
        self.factor = factor
        self.p = p

    def __call__(self, data: Data) -> Data:
        if random.random() > self.p:
            return data

        if hasattr(data, self.attribute):
            data[self.attribute] = data[self.attribute] * self.factor

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.attribute}, p={self.p})"
