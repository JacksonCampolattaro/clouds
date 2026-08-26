import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool
from torch_geometric.transforms import BaseTransform


class ExtractHeights(BaseTransform):
    def __init__(self, gravity_axis: int = 2, ground: bool = False, scale: float | None = None):
        super().__init__()
        self.gravity_axis = gravity_axis
        self.ground = ground
        self.scale = scale

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)
        data.height = data.pos[:, self.gravity_axis].unsqueeze(-1)

        if self.ground:
            assert not isinstance(data.batch, Tensor)
            data.height = data.height - torch.amin(data.height)

        if self.scale:
            data.height = data.height * self.scale

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(gravity_axis={self.gravity_axis}, ground={self.ground})'


class ExtractCoords(BaseTransform):
    def __init__(self, dims: list[int] | None = None, ground: bool = False, scale: float | None = None):
        super().__init__()
        self.dims = dims
        self.ground = ground
        self.scale = scale

    def forward(self, data: Data) -> Data:
        assert isinstance(data.pos, Tensor)
        data.coord = data.pos[:, self.dims] if self.dims else data.pos

        if self.ground:
            assert not isinstance(data.batch, Tensor)
            data.coord = data.coord - torch.amin(data.height)

        if self.scale:
            data.coord = data.coord * self.scale

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dims={self.dims}, ground={self.ground}, scale=self.scale)'
