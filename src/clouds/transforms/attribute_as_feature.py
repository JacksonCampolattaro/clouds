import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AttributeAsFeature(BaseTransform):
    def __init__(
        self,
        attributes: list[str],
        overwrite: bool = True,
        drop: bool = True,
    ):
        super().__init__()
        self.attributes = attributes
        self.overwrite = overwrite
        self.drop = drop

    def forward(self, data: Data) -> Data:
        assert isinstance(data.num_nodes, int)

        if not isinstance(data.x, Tensor) or self.overwrite:
            data.x = torch.zeros([data.num_nodes, 0], dtype=data.pos.dtype, device=data.pos.device)

        for key in self.attributes:
            assert hasattr(data, key) and isinstance(data[key], Tensor)
            assert isinstance(data.x, Tensor)
            data.x = torch.cat([data.x, data[key].to(dtype=data.x.dtype)], dim=-1)
            if self.drop:
                del data[key]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attributes={self.attributes}, overwrite={self.overwrite}, drop={self.drop})"
