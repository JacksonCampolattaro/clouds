import random

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AttributeDropout(BaseTransform):
    def __init__(self, feature: str, p=0.2):
        super().__init__()
        self.feature = feature
        self.p = p

    def forward(self, data: Data) -> Data:
        if random.random() > self.p:
            return data

        if hasattr(data, self.feature):
            data[self.feature].fill_(0)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.feature}, p={self.p})"
