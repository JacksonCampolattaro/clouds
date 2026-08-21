from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Identity(BaseTransform):
    def forward(self, data: Data) -> Data:
        return data
