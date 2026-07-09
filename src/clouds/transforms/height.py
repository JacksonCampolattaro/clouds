import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool
from torch_geometric.transforms import BaseTransform


class ExtractHeights(BaseTransform):
    def __init__(self, gravity_axis: int = 2, ground: bool = False):
        super().__init__()
        self.gravity_axis = gravity_axis
        self.ground = ground

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)
        data.height = data.pos[:, self.gravity_axis].unsqueeze(-1)

        if self.ground:
            if 'batch' in data:
                min_heights = -global_max_pool(-data.height, batch=data.batch)
                data.height = data.height - min_heights[data.batch]
            else:
                data.height = data.height - torch.amin(data.height)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(gravity_axis={self.gravity_axis}, ground={self.ground})'
