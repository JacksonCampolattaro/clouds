from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class Identity(BaseTransform):
    def __call__(self, data: Data) -> Data:
        return data
