from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class DiscardAttributes(BaseTransform):
    def __init__(self, *attributes: str):
        super().__init__()
        self.attributes = attributes

    def __call__(self, data: Data) -> Data:

        for attribute in self.attributes:
            if attribute == 'pos':
                data.num_nodes = data[attribute].size(0)
            del data[attribute]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.attributes})"
